"""
Written by: Tommy Banker (thomas_banker@berkeley.edu)
"""

import os
import time
import json
from utils import *
import numpy as np

import botorch
#botorch.settings.suppress_botorch_warnings(True)
import warnings

warnings.simplefilter("ignore" )




def safe_eval(f, x):
  n = len(x)
  nm = 1000
  splits = int(np.ceil(n/nm))
  obs = []
  for i in range(splits):
    if i-1 != splits:
      xi = x[nm*i:nm*(i+1)]
    else:
      xi = x[nm*i:]
    obs.append(f(xi).tolist())
  obs = torch.tensor(sum(obs, []))
  return obs
  
  
def qei (train_x, train_y, train_smiles,test_x, test_y, test_smiles, q_samples, exp_config):
	train_x0, train_y0, train_smiles0 = train_x, train_y, train_smiles
	test_x0, test_y0, test_smiles0    = test_x, test_y, test_smiles
	dim = test_x.shape[-1]
	next_sample_index = []
	
	for i in range(q_samples):
		keep_idx = np.arange(train_x.shape[-1])
		dim = test_x.shape[-1]
		
		try : del model0
		except: a = 0
	
		model0,mll0 = create_model(train_x0, train_y0, keep_idx, exp_config)
		model0,mll0 = fit_model(model0, mll0, train_x0, train_y0, exp_config)
		
		f_acq = ExpectedImprovement(model0, best_f=max(train_y0))
		with torch.no_grad():
			acq = safe_eval(f_acq, test_x0.reshape(-1,1,dim))
			max_acq = torch.max(acq)
			max_acq_index = torch.argmax(acq).item()      
			next_sample_index.append(max_acq_index)
			pos = model0.posterior(test_x0[max_acq_index,:].reshape(1,1,-1))
			mean, var = pos.mean.mean(),pos.variance.mean()

	z = torch.randn(1)
	fantasy = mean + (var**0.5)*z
	
	del mll0
	del f_acq
	
	next_sample_index.append(max_acq_index)
	train_x0, train_y0, train_smiles0, test_x0, test_y0, test_smiles0 = update_test_train_data(np.array(max_acq_index).flatten(), train_x0, train_y0, train_smiles0, test_x0, test_y0, test_smiles0)
	train_y0[-1] = fantasy

	return next_sample_index, model0



def run_config(config, run=None, q=None):
    '''
    Runs a BO experiment based on the input configuration file.
    args:
        config (string): .json file location relative to main.py
    '''
    
    print('Loading configuration')
    with open(config) as f:
        exp_config = json.load(f)
    f.close()

    if run is not None: exp_config['seed'] = run
    if q is not None: exp_config['bo']['acq_fun_params']['q'] = q

    if not os.path.exists(exp_config['save_dir']):
        os.mkdir(exp_config['save_dir'])   
    save_dir = os.path.join(exp_config['save_dir'], exp_config['exp_name'])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    seed_exp(exp_config)
    print(save_dir)

    print('########### Gathering Data ###########')
    descr_df, smiles = gather_data(exp_config)

    y_variable = exp_config['common']['y_variable']
    print('########### Processing Data ###########')
    xdf, ydf = format_data(descr_df, exp_config)
    del descr_df
    dim_red_method = exp_config['dim_red']['method']
    if dim_red_method == 'pca':
        xdf = pca_data(xdf, exp_config)

    train_x, train_y, train_smiles, test_x, test_y, test_smiles = split_data(xdf, ydf, smiles, exp_config)
    del ydf
    del smiles

    if dim_red_method == 'corr_thin':
        keep_idx, drop_idx = corr_thin_data(train_x, exp_config)
    elif dim_red_method == 'full_corr_thin':
        keep_idx, drop_idx = corr_thin_data(torch.cat([train_x, test_x]), exp_config)
    else:
        keep_idx = np.arange(train_x.shape[-1])
    print(f'Initial Training Set Size: {len(train_y)} \nDimensionality: {len(keep_idx)}')

    result_start_time = time.time()
    iter_start_time = time.time()

    model_type = exp_config['common']['model_type']
    #print('########### Making Model ###########')
    #model, mll = create_model(train_x, train_y, keep_idx, exp_config)
    #model, mll = fit_model(model, mll, train_x, train_y, exp_config)

    '''
    print('########### Saving Data ###########')
    final_iter = False
    filename = get_filename(0, exp_config)
    print(filename)
    save_iter(filename, save_dir, iter_start_time, model,
              xdf, train_x, train_y, train_smiles, keep_idx,
              final_iter, exp_config)'''

    
    init_budget = exp_config['common']['init_budget']
    total_budget = exp_config['common']['total_budget']
    bo_iters = np.arange(1, total_budget - init_budget + 1)
    acq_fun = exp_config['bo']['acq_fun']
    acq_fun_params = exp_config['bo']['acq_fun_params']
    final_iter = False
    
    
    try: q_samples = acq_fun_params['q']
    except: q_samples = 1


    for bo_iter in range(int(np.ceil(bo_iters[-1]/q_samples))):
        print('########### Starting BO Iter: ' + str(bo_iter) + ' ########### - q = '+str(q_samples))
        iter_start_time = time.time()

        if acq_fun == "qEI":
            next_sample_index, model = qei(train_x, train_y, train_smiles, test_x, test_y, test_smiles, q_samples,exp_config)
        else:
            model, mll = create_model(train_x, train_y, keep_idx, exp_config)
            model, mll = fit_model(model, mll, train_x, train_y, exp_config)
            next_sample_index = get_sample(model,mll, test_x,test_y, test_smiles, train_x, train_y,train_smiles, keep_idx, exp_config, bo_iter)

        train_x, train_y, train_smiles, test_x, test_y, test_smiles = update_test_train_data(np.array(next_sample_index).flatten(), train_x, train_y, train_smiles, test_x, test_y, test_smiles)
        print(f'Acquired Sample! ')
        print(train_smiles[-q_samples:])
        
        
        if bo_iter + init_budget == np.ceil(bo_iters[-1]/q_samples):
            final_iter = True
        filename = get_filename(bo_iter, exp_config)
        save_iter(filename, save_dir, iter_start_time, model,
                  xdf, train_x, train_y, train_smiles, keep_idx,
                  final_iter, exp_config)
        print('Saving Data!')

        if dim_red_method == 'corr_thin' and 'thin_freq' != None:
            print('Rethinning')
            thin_freq = exp_config['dim_red']['thin_freq']
            if bo_iter % thin_freq == 0:
                keep_idx, drop_idx = corr_thin_data(train_x, exp_config)


if __name__ == '__main__':   

	configs1 = [r'./config/malaria/Malaria_SAAS_0.json',
				#r'./config/malaria/Malaria_GP_full_0.json',
				#r'./config/malaria/Malaria_GP_pca_0.json',
				#r'./config/malaria/Malaria_linear_0.json',
				#r'./config/malaria/Malaria_Random_0.json'
				]

	configs2 = [r'./config/cep/CEP_SAAS_0.json',
				#r'./config/cep/CEP_GP_full_0.json',
				#r'./config/cep/CEP_GP_pca_0.json',
				#r'./config/cep/CEP_Linear_0.json',
				#r'./config/cep/CEP_Random_0.json'
				]
							
	configs3 = [#r'./config/lipo/lipophilicity_GP_full_0.json',
				#r'./config/lipo/lipophilicity_GP_pca_0.json',
				#r'./config/lipo/lipophilicity_linear_0.json',
				r'./config/lipo/lipophilicity_Random_0.json'
				#r'./config/lipo/lipophilicity_SAAS_0.json'
				]
							

	
	#configs4 = [r'./config/cep_q/CEP_Random_0.json',]
	#configs4 = [r'./config/dg/dg_SAAS_0.json']
	configs4 = [r'./config/dg/dg_GP_full_0.json',
				r'./config/dg/dg_GP_pca_0.json',
				r'./config/dg/dg__Random_0.json',
				r'./config/dg/dg_SAAS_0.json']

	feat = ''				
	configs = [#f'./config/logp{feat}/logp_GP_full_0.json',
				#f'./config/logp{feat}/logp_GP_pca_0.json',
				#f'./config/logp{feat}/logp_Random_0.json',
				f'./config/logp{feat}/logp_SAAS_0.json',
				]


	#configs = configs4
	qlist = [1]
	for q in qlist:
	    for i, config in enumerate(configs):
		    for j in range(5):
			    print(j)
			    run_config(config,run=j, q=q)#''''''




