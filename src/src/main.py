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

	configs = [#r'./config/malaria/Malaria_SAAS_0.json',
				r'./config/malaria/Malaria_GP_full_0.json',
				]



	for i, config in enumerate(configs):
		for j in range(2):
			print(j)
			run_config(config,run=j)#''''''




