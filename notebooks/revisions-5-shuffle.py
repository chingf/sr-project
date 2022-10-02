import numpy as np
from scipy.io import loadmat
from scipy.stats import gamma
from scipy.special import digamma, loggamma
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
import os
import sys
from joblib import Parallel, delayed
root = os.path.dirname(os.path.abspath(os.curdir))
sys.path.append(root)
from datasets import inputs, sf_inputs_discrete  
import configs

iters = 3                                                                      
n_jobs = 56
root_dir = "../../engram/Ching/03_hannah_dset_revisions/"

def collect_metrics(arg):
    gammas = []
    spatial_sigmas = []
    sparsities = []
    errors = []
    gamma, sigma, sparsity = arg
    gamma_dir = f'{root_dir}sparsity{sparsity}/sigma{sigma}/{gamma}/'
    linear_model_dir = f'{gamma_dir}linear/'
    for _iter in os.listdir(linear_model_dir):
        linear_iter_dir = f'{linear_model_dir}/{_iter}/'
        results_path = linear_iter_dir + 'results.p'
        if not os.path.isfile(results_path): continue
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        dset = results['dset']
        linear_net = results['net']
        linear_M = linear_net.M.numpy().squeeze()
        linear_M = np.random.choice(linear_M.flatten(), size=linear_M.shape)
        inputs = dset.dg_inputs.T
        outputs = dset.dg_inputs.T @ linear_M
        val = inputs[1:] + gamma*outputs[:-1]
        expected = outputs[1:]
        error = np.mean(np.square(val - expected), axis=1)
        error = np.mean(error)
        gammas.append(gamma)
        spatial_sigmas.append(sigma)
        sparsities.append(dset.feature_maker.post_smooth_sparsity)
        errors.append(error)
    return gammas, spatial_sigmas, sparsities, errors

# Arguments
gammas = [0.4, 0.6, 0.8]
spatial_sigmas = [0.0, 1.0, 2.0, 3.0]                                          
sparsity_range = [[0.001, 0.2], [0.001, 0.1], [0.001, 0.04], [0.001, 0.023]]   
spatial_sigmas.extend([0.25, 0.5, 1.25, 1.5, 1.75, 2.25, 2.5, 2.75, 3.25])     
sparsity_range.extend([                                                        
    [0.001, 0.19], # 0.25                                                      
    [0.001, 0.15], # 0.5                                                       
    [0.001, 0.09], # 1.25                                                      
    [0.001, 0.05], # 1.5                                                       
    [0.001, 0.045], # 1.75                                                     
    [0.001, 0.037], # 2.25                                                     
    [0.001, 0.03], # 2.5                                                       
    [0.001, 0.025], # 2.75                                                     
    [0.001, 0.021], # 3.25                                                     
    ])                                                                         
num_states = 14*14                                                             
num_steps = 5001
args = []                                                                   
for gamma in gammas:                                                        
    for idx, spatial_sigma in enumerate(spatial_sigmas):                    
        _range = sparsity_range[idx]                                        
        sparsity_ps = np.linspace(
            _range[0], _range[1],
            num=20, endpoint=True)
        for sparsity_p in sparsity_ps:                                      
            args.append([gamma, spatial_sigma, sparsity_p]) 

gammas = []
spatial_sigmas = []
sparsities = []
errors = []
job_results = Parallel(n_jobs=n_jobs)(delayed(collect_metrics)(arg) for arg in args)
for res in job_results:
    gammas.extend(res[0])
    spatial_sigmas.extend(res[1])
    sparsities.extend(res[2])
    errors.extend(res[3])

df = pd.DataFrame({
    'Gamma': gammas,
    'Sigma': spatial_sigmas,
    'Sparsity': sparsities,
    'Error': errors
})

with open(f'{root_dir}shuffle_td_errors.p', 'wb') as f:
    pickle.dump(df, f)
