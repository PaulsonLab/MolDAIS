"""
Written by: Tommy Banker (thomas_banker@berkeley.edu)
"""

import os
import time
import random
import numpy as np
import pandas as pd
import torch

from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch import fit_gpytorch_model
from botorch import fit_fully_bayesian_model_nuts
from botorch.models.transforms import Standardize
from botorch.models.transforms import Normalize
from torch import corrcoef
from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound
from botorch.acquisition import qExpectedImprovement, qUpperConfidenceBound


import pdb
#from botorch.acquisition.analytic import qlogExpectedImprovement



def seed_exp(exp_config):
    '''
    Seeds the BO experiment based on the input configuration file.
    args:
        exp_config (dict): dictionary of BO experiment parameters
    '''
    random.seed(exp_config['seed'])
    np.random.seed(exp_config['seed'])
    torch.manual_seed(exp_config['seed'])

def gather_data(exp_config):
    '''
    Gathers molecular descriptor data from .csv of molecular descriptors.
    args:
        exp_config (dict): dictionary of BO experiment parameters

    returns:
        descr_df (pd.DataFrame): dataframe containing "x" and "y" variables
        smiles (pd.DataFrame): dataframe of molecules' SMILES representations
    '''
    # load data
    data_loc = exp_config['data_loc']    
    try: df = pd.read_csv(data_loc, index_col=False)
    except: pdb.set_trace()
    smiles = df['SMILES']
    # drop any irrelevent properties
    drop_props = exp_config["common"]["drop_props"]
    if drop_props != None:
        df = df.drop(columns=drop_props)
    # remove non-numeric columns
    df2 = df.select_dtypes(include=['float64','int64'] )
    # remove zero-varience columns 
    dfx = list(df2.var())
    descr_df = df2.iloc[:,[dfx[i]>0 for i in range(len(dfx))]]
    
    return descr_df, smiles




def format_data(df, exp_config):
    '''
    Separates the input df into an "x" and "y" dataframe.
    args:
        df (pd.DataFrame): dataframe containing "x" and "y" variables
        exp_config (dict): dictionary of BO experiment parameters

    returns:
        xdf (pd.DataFrame): dataframe containing "x" variables
        ydf (pd.DataFrame): dataframe containing "y" variable
    '''
    y_variable = exp_config['common']['y_variable']

    ydf = df[y_variable]
    
    if 'Gsolv' in df.columns or 'E' in df.columns:
        try: xdf = df.drop(columns=['Gsolv'])
        except:  xdf = df.drop(columns=['E'])
    else:
        xdf = df.drop(columns=[y_variable])
        
    return xdf, ydf





def split_data(xdf, ydf, smiles, exp_config):   
    '''
    Separates the input df into an "x" and "y" dataframe.
    args:
        xdf (pd.DataFrame): dataframe containing "x" variables
        ydf (pd.DataFrame): dataframe containing "y" variable
        smiles (pd.DataFrame): dataframe of molecules' SMILES representations
        exp_config (dict): dictionary of BO experiment parameters

    returns:
        train_x (torch.Tensor): tensor of "x" variables for sampled molecules
        train_y (torch.Tensor): tensor of "y" variable for sampled molecules
        train_smiles (pd.DataFrame): dataframe of sampled molecules' SMILES representations
        test_x (torch.Tensor): tensor of "x" variables for unsampled molecules
        test_y (torch.Tensor): tensor of "y" variable for unsampled molecules
        test_smiles (pd.DataFrame): dataframe of unsampled molecules' SMILES representations
    '''
    # get indices to split data into sampled and unsampled (test) sets
    if exp_config['init_ind_loc'] != None:
        sample_indices = np.load(exp_config['init_ind_loc'])
    else:
        init_budget = exp_config['common']['init_budget']
        sample_indices = np.random.choice(np.arange(len(ydf)), init_budget, replace=False)
    test_indices = np.setdiff1d(np.arange(len(xdf)), sample_indices, assume_unique=True)
    
    # split based on indices & convert df into tensor  
    train_x = torch.tensor(xdf.iloc[sample_indices].values).float()
    train_y = torch.tensor(ydf.iloc[sample_indices].values).unsqueeze(1).float()
    test_x = torch.tensor(xdf.iloc[test_indices].values).float()
    test_y = torch.tensor(ydf.iloc[test_indices].values).unsqueeze(1).float()
    train_smiles = smiles.iloc[sample_indices].to_numpy()
    test_smiles = smiles.iloc[test_indices].to_numpy()        
    
    return train_x, train_y, train_smiles, test_x, test_y, test_smiles

def pca_data(xdf, exp_config):
    '''
    Use PCA to reduce the dimensionality of "x" variables to desired dimensions. 
    args:
        xdf (pd.DataFrame): dataframe containing "x" variables
        exp_config (dict): dictionary of BO experiment parameters
    
    returns:
        xdf (pd.DataFrame): dataframe containing reduced dimension "x" variables
    '''
    pca_components = exp_config['dim_red']['n_dims']
    pca = PCA(pca_components)
    principal_components = pca.fit_transform(xdf)
    dfPrincipal = pd.DataFrame(data = principal_components,
                                columns = [('PC' + str(x)) for x in range(1, pca_components+1)])
    xdf = dfPrincipal
    
    return xdf

def corr_thin_data(train_x, exp_config):
    '''
    Calculates correlation between "x" variables and drops variables highly correlated with others 
    args:
        train_x (torch.Tensor): tensor of "x" variables for sampled molecules
        exp_config (dict): dictionary of BO experiment parameters
    
    returns:
        keep_idx (np.array): array of indices corresponding to "x" variables
            comprising dimensinoality reduced latent space
        drop_idx (np.array): array of indices corresponding to "x" variables
            excluded from the dimensinoality reduced latent space
    '''
    corr_threshold = exp_config['dim_red']['corr_thresh']

    # compute correlation matrix for data - taking absolute and ignoring correlations on/below diagonal
    corr = train_x.T.corrcoef()
    corr = torch.absolute(torch.triu(corr, diagonal=1)) # set elements on/below diagonal to 0

    # find correlated features and loop through them - dropping the 2nd dimension if not already
    dims1, dims2 = torch.where(corr > corr_threshold)
    drop_idx = torch.LongTensor()
    for dim1, dim2 in zip(dims1, dims2):
        if dim2 not in drop_idx:
            drop_idx = torch.cat((drop_idx, dim2.unsqueeze(0)), 0)
    keep_idx = np.setdiff1d(np.arange(train_x.shape[-1]), drop_idx)
    
    return keep_idx, drop_idx

def create_model(train_x, train_y, keep_idx, exp_config):
    '''
    Creates an unfitted surrogate model relating "x" variables to "y" variable
    args:
        train_x (torch.Tensor): tensor of "x" variables for sampled molecules
        train_y (torch.Tensor): tensor of "y" variable for sampled molecules
        keep_idx (np.array): array of indices corresponding to "x" variables
            comprising dimensinoality reduced latent space (if using correlation thinning)
        exp_config (dict): dictionary of BO experiment parameters
    
    returns:
        model (GPyTorch.model or sklearn.linear_model): unfitted surrogate model
        mll (GPyTorch.mll): marginal likelihood for Gaussian process model
    '''
    model_type = exp_config['common']['model_type']
    model = None
    mll = None
    
    # create Gaussian process model
    if model_type == 'GP':
        model = SingleTaskGP(train_X=train_x, train_Y=train_y, input_transform=Normalize(d=train_x.size(1)), outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
        
    # create linear model
    elif model_type == 'Linear':
        model = linear_model.LinearRegression()

    # create SAAS Gaussian process model with option of only using features remaining after correlation thinning
    elif model_type == 'SAASGP':
        if exp_config['dim_red']['method'] == 'corr_thin':
            model = SaasFullyBayesianSingleTaskGP(train_X=train_x[:,keep_idx], train_Y=train_y, input_transform=Normalize(d=train_x[:,keep_idx].size(1)), outcome_transform=Standardize(m=1))  
        else:
            model = SaasFullyBayesianSingleTaskGP(train_X=train_x, train_Y=train_y, input_transform=Normalize(d=train_x.size(1)), outcome_transform=Standardize(m=1))  


    return model, mll
    
    
    
    

def fit_model(model, mll, train_x, train_y, exp_config):
    '''
    Fit the surrogate model to the sampled molecules' training data
    args:
        model (GPyTorch.model or sklearn.linear_model): unfitted surrogate model
        mll (GPyTorch.mll): marginal likelihood for Gaussian process model
        train_x (torch.Tensor): tensor of "x" variables for sampled molecules
        train_y (torch.Tensor): tensor of "y" variable for sampled molecules
        exp_config (dict): dictionary of BO experiment parameters
    
    returns:
        model (GPyTorch.model or sklearn.linear_model): fitted surrogate model
        mll (GPyTorch.mll): marginal likelihood for Gaussian process model
    '''
    model_type = exp_config['common']['model_type']
       
    # fit Gaussian process model
    if model_type == 'GP':
        fit_gpytorch_model(mll)

    # fit linear model      
    elif model_type == 'Linear':
        Xscaler = MinMaxScaler().fit(train_x)
        Yscaler = StandardScaler().fit(train_y)
        model.fit(Xscaler.transform(train_x),
                  Yscaler.transform(train_y))

    # fit SAAS Gaussian process model with option of only using features remaining after correlation thinning           
    elif model_type == 'SAASGP':
        WARMUP_STEPS = exp_config['bo']['saas_fit_params']['WARMUP_STEPS']
        NUM_SAMPLES = exp_config['bo']['saas_fit_params']['NUM_SAMPLES']
        THINNING = exp_config['bo']['saas_fit_params']['THINNING']
        
        fit_fully_bayesian_model_nuts(model, warmup_steps=WARMUP_STEPS, num_samples=NUM_SAMPLES, thinning=THINNING, disable_progbar=False)
    return model, mll


def get_filename(bo_iter, exp_config):
    '''
    Produce filename for given experiment and iteration within the BO experiment
    args:
        bo_iter (int): iteration within the BO experiment
        exp_config (dict): dictionary of BO experiment parameters
    
    returns:
        filename (string): specially formatted string to reflect BO experiment parameters
    '''
    # record key configuration details/parameters
    exp_name = exp_config['exp_name']
    seed = exp_config['seed']
    min_max = exp_config['common']['max_min']
    y_variable = exp_config['common']['y_variable']
    model_type = exp_config['common']['model_type']
    init_budget = exp_config['common']['init_budget']
    total_budget = exp_config['common']['total_budget']
    acq_fun = str(exp_config['bo']['acq_fun'])
    dim_red_method = str(exp_config['dim_red']['method'])
    #print(exp_config)
    
    #q_samples = exp_config['bo']['acq_fun_params']['q']
    
    if dim_red_method == 'pca':
         dim_red_param = exp_config['dim_red']['n_dims'] 
    elif dim_red_method == 'corr_thin':
         dim_red_param = exp_config['dim_red']['corr_thresh'] 
    else:
        dim_red_param = 'None'
        dim_red_method = 'None'
        
    # create filename to reflect configuration details/parameters
    #import pdb; pdb.set_trace()

    '''# with batching in exp_config
    filename = '_'.join([exp_name, min_max, y_variable,                         
				model_type,'q',str(q_samples), 'seed', str(seed), 'bo-iter', str(bo_iter),                         
				str(init_budget), 'to', str(total_budget), 'samples',                         
				str(acq_fun), dim_red_method, str(dim_red_param)])

    '''
    filename = '_'.join([exp_name, min_max, y_variable,                         
				model_type, 'seed', str(seed), 'bo-iter', str(bo_iter),                         
				str(init_budget), 'to', str(total_budget), 'samples',                         
				str(acq_fun), dim_red_method, str(dim_red_param)])				
				
    
    return filename

def save_iter(filename, save_dir, iter_start_time, model, xdf, train_x, train_y, train_smiles, keep_idx, final_iter, exp_config):
    '''
    Save training and model information
    args:
        filename (string): specially formatted string to reflect BO experiment parameters
        save_dir (string): (string): file loc to save .pkl of experiment
        iter_start_time (float): time of start of experiment iteration
        model (GPyTorch.model or sklearn.linear_model): fitted surrogate model
        xdf (pd.DataFrame): dataframe containing reduced dimension "x" variables
        train_x (torch.Tensor): tensor of "x" variables for sampled molecules
        train_y (torch.Tensor): tensor of "y" variable for sampled molecules
        train_smiles (pd.DataFrame): dataframe of sampled molecules' SMILES representations
        keep_idx (np.array): array of indices corresponding to "x" variables
            comprising dimensinoality reduced latent space
        final_iter (bool): if this is the final iteration of the experiment
        exp_config (dict): dictionary of BO experiment parameters
    '''
    model_type = exp_config['common']['model_type']

    if final_iter == False:
        # for non-final iterations, save the lengthscales/coefficients and the iteration time
        if model_type == 'GP':
            torch.save({"iter_time":(time.time() - iter_start_time)
                        }, os.path.join(save_dir, filename + ".pkl"))
        elif model_type == 'Linear':
            torch.save({"coefficients":torch.from_numpy(model.coef_[0]),
                        "dimensions":xdf.columns.to_list(),
                        "iter_time":(time.time() - iter_start_time)
                        }, os.path.join(save_dir, filename + ".pkl"))
        elif model_type == 'SAASGP':
            if exp_config['dim_red']['method'] == 'corr_thin':
                lengthscales = np.zeros(xdf.shape[-1])
                lengthscales[keep_idx] = model.median_lengthscale.detach()
                torch.save({"lengthscales":lengthscales,
                            "dimensions":xdf.columns.to_list(),
                            "iter_time":(time.time() - iter_start_time)
                            }, os.path.join(save_dir, filename + ".pkl"))
            else:
                torch.save({"lengthscales":model.median_lengthscale.detach(),
                            "dimensions":xdf.columns.to_list(),
                            "iter_time":(time.time() - iter_start_time)
                            }, os.path.join(save_dir, filename + ".pkl"))
    else:
        # for final iterations, save the training data (and SMILES) along with lengthscales/coefficients and the iteration time
        if model_type == 'GP':
            torch.save({"y_train":train_y,
                        "x_train":train_x,
                        "smiles_train":train_smiles.squeeze(),
                        "dimensions":xdf.columns.to_list(),
                        "iter_time":(time.time() - iter_start_time)
                        }, os.path.join(save_dir, filename + ".pkl"))
        elif model_type == 'Linear':
            torch.save({"y_train":train_y,
                        "x_train":train_x,
                        "smiles_train":train_smiles.squeeze(),
                        "coefficients":torch.from_numpy(model.coef_[0]),
                        "dimensions":xdf.columns.to_list(),
                        "iter_time":(time.time() - iter_start_time)
                        }, os.path.join(save_dir, filename + ".pkl"))
        elif model_type == 'SAASGP':
            # if using correlation thinning, only save the features used in the model
            if exp_config['dim_red']['method'] == 'corr_thin':
                lengthscales = np.zeros(train_x.shape[-1])
                lengthscales[keep_idx] = model.median_lengthscale.detach()
                torch.save({"y_train":train_y,
                            "x_train":train_x,
                            "smiles_train":train_smiles.squeeze(),
                            "lengthscales":lengthscales,
                            "dimensions":xdf.columns.to_list(),
                            "iter_time":(time.time() - iter_start_time)
                            }, os.path.join(save_dir, filename + ".pkl"))
            else:
            # if not using correlation thinning, save all the features used in the model
                torch.save({"y_train":train_y,
                            "x_train":train_x,
                            "smiles_train":train_smiles.squeeze(),
                            "lengthscales":model.median_lengthscale.detach(),
                            "dimensions":xdf.columns.to_list(),
                            "iter_time":(time.time() - iter_start_time)
                            }, os.path.join(save_dir, filename + ".pkl"))




def get_sample(model,mll, test_x,test_y,test_smiles, train_x,train_y,train_smiles, keep_idx, exp_config,bo_iter):
    '''
    Acquire next sample from unsampled molecules with its property value
    args:
        model (GPyTorch.model or sklearn.linear_model): fitted surrogate model
        test_x (torch.Tensor): tensor of "x" variables for unsampled molecules
        train_y (torch.Tensor): tensor of "y" variable for sampled molecules
        keep_idx (np.array): array of indices corresponding to "x" variables
            comprising dimensinoality reduced latent space
        exp_config (dict): dictionary of BO experiment parameters

    returns:
        next_sample_index (int): index of sampled molecule within test_x
    '''
    
    init_budget = exp_config['common']['init_budget']
    model_type = exp_config['common']['model_type']
    acq_fun = exp_config['bo']['acq_fun']
    acq_fun_params = exp_config['bo']['acq_fun_params']
    dim_red_method = exp_config['dim_red']['method']
    q_samples = 1
    if dim_red_method == 'pca' or model_type == 'Linear':
        n_dims = exp_config['dim_red']['n_dims']
    elif dim_red_method == 'corr_thin':
        test_x = test_x[:,keep_idx]
    
    dim = test_x.shape[1]
    with torch.no_grad():

        if acq_fun == 'random':
            # acquire sample at random from unsampled set
            random.seed((exp_config['seed']+1)**bo_iter   )
            np.random.seed(exp_config['seed'])
            next_sample_index = np.random.choice(np.arange(test_x.shape[0]), replace=False)
            
            
        elif acq_fun == 'qrandom':
            q_samples = acq_fun_params['q']
            # acquire sample at random from unsampled set
            random.seed((exp_config['seed']+1)**bo_iter   )
            np.random.seed(exp_config['seed'])
            next_sample_index = np.random.choice(np.arange(test_x.shape[0]), q_samples,replace=False)

            #import pdb
            #pdb.set_trace()
            

        elif model_type == 'GP':
            try: q_samples = acq_fun_params['q']
            except: q_samples = 1
            # acquire sample based on acq_fun and GP posterior from unsampled set
            if acq_fun == 'EI':
                f_acq = ExpectedImprovement(model, best_f=max(train_y))
		        # breaking acquisition evaluatiosn into chunks of 1000 molecules due to hardware limits
                max_acq = 0
                max_acq_index = 0
                group_size = 1000
                n_groups = np.int64(np.ceil(test_x.shape[0] / group_size))
                for group in np.arange(n_groups):
                    group_index = group*group_size
                    if group < (n_groups-1):
                        acq = f_acq(test_x[group_index:(group_index+group_size)].reshape(group_size,1,dim))
                    else:
                        acq = f_acq(test_x[group_index:].reshape((test_x.shape[0]-group_index),1,dim))
                    if torch.max(acq) > max_acq:
                        max_acq = torch.max(acq)
                        max_acq_index = torch.argmax(acq) + group_index
                next_sample_index = max_acq_index

            elif acq_fun == 'UCB':
                f_acq = UpperConfidenceBound(model, beta=acq_fun_params['beta'])
                max_acq = 0
                max_acq_index = 0
                group_size = 1000
                n_groups = np.int64(np.ceil(test_x.shape[0] / group_size))
                for group in np.arange(n_groups):
                    group_index = group*group_size
                    if group < (n_groups-1):
                        acq = f_acq(test_x[group_index:(group_index+group_size)].reshape(group_size,1,dim))
                    else:
                        acq = f_acq(test_x[group_index:].reshape((test_x.shape[0]-group_index),1,dim))
                    if torch.max(acq) > max_acq:
                        max_acq = torch.max(acq)
                        max_acq_index = torch.argmax(acq) + group_index
                next_sample_index = max_acq_index



            
        elif model_type == 'Linear':
            # acquire sample based max prediction of linear model using only top n_dims
            # find the largest feature coefficients
            principal_features = np.argsort(np.absolute(model.coef_))[0][-1*n_dims:]
            principal_coef = model.coef_[0][principal_features]
            # only look at data corresponding to "top" features
            principal_data = test_x[:,principal_features.tolist()]
            # compute predictions using only "top" components to find max prediction
            predictions = (principal_data * principal_coef).sum(axis=1)
            next_sample_index = np.argmax(predictions)
                            
        elif model_type == 'SAASGP':
            #q_samples = acq_fun_params['q']
            # acquire sample based on acq_fun and SAASGP posterior from unsampled set
            if acq_fun == 'EI'  :
                f_acq = qExpectedImprovement(model, best_f=max(train_y), )
                acq = f_acq(test_x.reshape(-1,1,dim))
                max_acq = torch.max(acq)
                max_acq_index = torch.argmax(acq)      
                next_sample_index = max_acq_index
            elif acq_fun == 'qUCB':
                f_acq = qUpperConfidenceBound(model, beta=acq_fun_params['beta'])
                acq = f_acq(test_x.reshape(-1,1,dim))
                max_acq = torch.max(acq)
                max_acq_index = torch.argmax(acq)      
                next_sample_index = max_acq_index

            elif acq_fun == 'qAUCB':
                f_acq = qAdaptiveUpperConfidenceBound(model,
                                                      beta=acq_fun_params['beta'],
                                                      decay=acq_fun_params['decay'],
                                                      n=train_y.shape[0] - init_budget)
                acq = f_acq(test_x.reshape(-1,1,dim))
                max_acq = torch.max(acq)
                max_acq_index = torch.argmax(acq)      
                next_sample_index = max_acq_index
            elif acq_fun == 'qAEI':
                f_acq = qAbruptExpectedImprovement(model,
                                                  beta=acq_fun_params['beta'],
                                                  xi=acq_fun_params['xi'],
                                                  eta=acq_fun_params['eta'],
                                                  best_f=max(train_y),
                                                  past_f=train_y,
                                                  n=train_y.shape[0] - init_budget)
                acq = f_acq(test_x.reshape(-1,1,dim))
                max_acq = torch.max(acq)
                max_acq_index = torch.argmax(acq)      
                next_sample_index = max_acq_index
            else: 
                print('No Acquisition Function!')

    return next_sample_index




def update_test_train_data(next_sample_indexs, train_x, train_y, train_smiles, test_x, test_y, test_smiles):
    '''
    Adds molecule corresponding to next_sample_index to the training data
    args:    
        next_sample_index (int): index of sampled molecule within test_x
        train_x (torch.Tensor): tensor of "x" variables for sampled molecules
        train_y (torch.Tensor): tensor of "y" variable for sampled molecules
        train_smiles (pd.DataFrame): dataframe of sampled molecules' SMILES representations
        test_x (torch.Tensor): tensor of "x" variables for unsampled molecules
        test_y (torch.Tensor): tensor of "y" variable for unsampled molecules
        test_smiles (pd.DataFrame): dataframe of unsampled molecules' SMILES representations

    returns:
        train_x (torch.Tensor): updated tensor of "x" variables for sampled molecules
        train_y (torch.Tensor): updated tensor of "y" variable for sampled molecules
        train_smiles (pd.DataFrame): updated dataframe of sampled molecules' SMILES representations
        test_x (torch.Tensor): updated tensor of "x" variables for unsampled molecules
        test_y (torch.Tensor): updated tensor of "y" variable for unsampled molecules
        test_smiles (pd.DataFrame): updated dataframe of unsampled molecules' SMILES representations
    '''
    #next_sample_indexs = np.sort([next_sample_indexs])[::-1]
    for i,next_sample_index in enumerate(next_sample_indexs):
        try: next_sample_index=next_sample_index.item()
        except: continue
       
        # get "x" and "y" values for sample and its corresponding SMILES string
        next_sample_x = test_x[next_sample_index].reshape((1,test_x.shape[-1]))
        next_sample_y = test_y[next_sample_index].reshape((1,test_y.shape[-1]))
        next_sample_smiles = test_smiles[next_sample_index]
        
        # update sampled and unsampled datasets with new sample
        train_x = torch.cat((train_x, next_sample_x))
        train_y = torch.cat((train_y, next_sample_y))
        train_smiles = np.hstack([train_smiles, [next_sample_smiles]])

        test_x = torch.cat([test_x[:next_sample_index], test_x[next_sample_index+1:]]) 
        test_y = torch.cat([test_y[:next_sample_index], test_y[next_sample_index+1:]]) 
        test_smiles = np.hstack([test_smiles[:next_sample_index], test_smiles[next_sample_index+1:]]) 
   
    return train_x, train_y, train_smiles, test_x, test_y, test_smiles


