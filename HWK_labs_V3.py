import numpy as np
from tick.hawkes import SimuHawkesExpKernels
from datetime import datetime
# Notes for Jacopo:
''' 

'''

# Define the module path + import packages
import sys
module_path = '/Users/jacoponudo/Documents/thesis/src/HWK'
sys.path.append(module_path)
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from HWK_package.functions import *
from HWK_package.tools_HWK_V3 import *
from scipy import stats
import random 
from scipy.stats import chi2
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import kstest
from tqdm import tqdm
from tick.hawkes import SimuHawkesExpKernels
from scipy import optimize

# Set source
source_data='/Users/jacoponudo/Documents/thesis/data/voat/voat_labeled_data_unified.parquet'
root='/Users/jacoponudo/Documents/thesis/'
output=root+'src/HWK/output'
output_threads=root+'src/HWK/output/temporary'

dataset = pd.read_parquet(source_data)





# Prepare observatios
x_values = np.arange(0, 1, 0.0001)  #griglia
root_submission='0'
root = dataset[dataset['root_submission'] == root_submission]
root.sort_values(by='created_at', inplace=True)

# Replicate with a model the conversation among users
users_acticvity=root.groupby('user')['comment_id'].count().reset_index()
users_acticvity=users_acticvity[users_acticvity['comment_id']>2]['user']
root=root[root['user'].isin(users_acticvity)]
users=root['user'].unique()#.sample(10).unique()
root=root[root['user'].isin(users)].reset_index()






observed_data = np.array([np.datetime64(x.replace(tzinfo=None)).astype(np.int64) for x in root['created_at']])
start_conversation = np.datetime64(min(root['created_at']).replace(tzinfo=None))
end_conversation = np.datetime64(max(root['created_at']).replace(tzinfo=None))
ℋ_t = (observed_data - start_conversation.astype(np.int64)) / (end_conversation.astype(np.int64) - start_conversation.astype(np.int64))
root.loc[:, 'time'] = ℋ_t.copy()
ECDF = np.array(F(x_values, ℋ_t))










# Define parameters
n_users = len(users)
import random
import time

results=[]
lambda_grid =(np.arange(1.0, 10.0,1.0))
alpha_grid = np.concatenate(([0.0001], np.arange(1.0, 7.0, 1.0)))
beta_grid = np.arange(0.01, 0.2, 0.1)
theta_combinations = list(product(lambda_grid, alpha_grid, beta_grid))
random.shuffle(theta_combinations)


for l, alpha, beta in tqdm(theta_combinations, total=len(theta_combinations)):
    
    start_time = time.time()  # Memorizza il tempo di inizio dell'iterazione
    
    lambdas = [lamda] * n_users
    alphas = [[alpha] * n_users] * n_users
    betas = [[beta] * n_users] * n_users

    end_time = 1.0  # Simula per 10 secondi

    hawkes_process = SimuHawkesExpKernels(adjacency=alphas, decays=betas, baseline=lambdas, end_time=end_time, force_simulation=True, verbose=False)
    hawkes_process.simulate()
    mu_ks, sd_ks = metrix(hawkes_process, users)
    
    results.append({'root': root_submission, 'lambda': l, 'alpha': alpha, 'beta': beta, 'mu_ks': mu_ks, 'sd_ks': sd_ks})

    # Verifica se è trascorso più di un minuto, in tal caso interrompi l'iterazione
    if time.time() - start_time > 120:
        print("Iterazione interrotta: tempo limite superato.")
        continue
df = pd.DataFrame(results)
df.to_csv('/Users/jacoponudo/Documents/thesis/src/HWK/outputs/grid_search_Sintetizzatore.csv', index=False) 

