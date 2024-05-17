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
root = dataset[dataset['root_submission'] == root_submission].copy()
root.sort_values(by='created_at', inplace=True)

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

alpha_grid = np.concatenate(([0.0000001], np.arange(0.5, 5.0, 0.5)))
lambda_grid =(np.arange(1.0, 10.0,1.0))
beta_grid = np.arange(0.05, 0.6, 0.05)
p_grid=np.arange(0.05, 0.3, 0.05)

theta_combinations = list(product(lambda_grid, alpha_grid, beta_grid,p_grid))
random.shuffle(theta_combinations)

    
for l, alpha, beta,p in tqdm(theta_combinations, total=len(theta_combinations)):
    
    start_time = time.time()  # Memorizza il tempo di inizio dell'iterazione
    
    alphas = generate_weighted_adjacency_matrix(alpha,n_users , p)
    lambdas = np.random.exponential(scale=1/l, size=n_users)
    betas = [list(np.random.uniform(0, beta, n_users)) for _ in range(n_users)]


    end_time = 1.0  # Simula per 10 secondi

    hawkes_process = SimuHawkesExpKernels(adjacency=alphas, decays=betas, baseline=lambdas, end_time=end_time, force_simulation=True, verbose=False)
    hawkes_process.simulate()
    KS, pvalue = metrix(hawkes_process, users,root,ECDF)
    
    results.append({'root': root_submission, 'lambda': l, 'alpha': alpha, 'beta': beta,'p':p, 'KS': KS, 'pvalue': pvalue})

    # Verifica se è trascorso più di un minuto, in tal caso interrompi l'iterazione
    if time.time() - start_time > 120:
        print("Iterazione interrotta: tempo limite superato.")
        continue
df = pd.DataFrame(results)
df.to_csv('/Users/jacoponudo/Documents/thesis/src/HWK/outputs/grid_search_Sintetizzatore.csv', index=False) 












import numpy as np
import matplotlib.pyplot as plt

# Assuming ℋ_t and ℋ_t_simulated are arrays of event times

# Calculate cumulative distributions

hawkes_process = SimuHawkesExpKernels(adjacency=alphas, decays=betas, baseline=lambdas, end_time=end_time, force_simulation=True, verbose=False)
hawkes_process.simulate()



# Plot the cumulative distributions
plt.plot(x_values, cumulative_dist_ℋ_t, label='ℋ_t Cumulative Distribution')
plt.plot(x_values, cumulative_dist_ℋ_t_simulated, label='ℋ_t_simulated Cumulative Distribution')

# Add labels and legend
plt.xlabel('Event Time')
plt.ylabel('Cumulative Probability')
plt.legend()

# Show plot
plt.show()


