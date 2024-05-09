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
from HWK_package.hawkes import *
from scipy import stats
import random 
from scipy.stats import chi2
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot

# Set source
source_data='/Users/jacoponudo/Documents/thesis/data/voat/voat_labeled_data_unified.parquet'
root='/Users/jacoponudo/Documents/thesis/'
output=root+'src/HWK/output'
output_threads=root+'src/HWK/output/temporary'

dataset = pd.read_parquet(source_data)

dataset['is_toxic']=dataset['toxicity_score']>0.6
user_toxic_activity=dataset.groupby(['root_submission','user'])['is_toxic'].sum().reset_index()
user_toxic_activity[user_toxic_activity['is_toxic']>10]



dataset_user=dataset[dataset.user=='smokratez'].copy()
dataset_user.sort_values(by='created_at', inplace=True)
dataset_user.root_submission.value_counts()
dataset_user=dataset_user[dataset_user.root_submission=='1073880'].copy()


#RQ1: Could we use a hawkes process to model the inter arrival time of comments?

# Preparo i dati e fitto il modello 
observed_data = np.array([np.datetime64(x.replace(tzinfo=None)).astype(np.int64) for x in dataset_user['created_at']])
mean_lag=np.mean(np.diff(observed_data))
ℋ_t=observed_data-min(observed_data)+mean_lag
𝛉_exp=exp_mle(ℋ_t,max(ℋ_t)+mean_lag)

# Faccio il QQplot del modello
tsShifted = exp_hawkes_compensators(ℋ_t, 𝛉_exp)
iat = np.diff(np.insert(tsShifted, 0, 0))
qqplot(iat, dist=stats.expon, fit=True, line="45")
plt.show()

# Estraggo una metric per valutare il modello
slope, intercept = sm.OLS(iat, sm.add_constant(stats.expon.ppf((np.arange(1, len(iat) + 1) - 0.5) / len(iat)))).fit().params
print("Pendenza della linea di riferimento adattata:", slope)
slope, intercept = 1, 0
predicted_values = slope * np.arange(1, len(iat) + 1) + intercept

# Calcolo delle distanze
distances = np.abs(iat - predicted_values)

print('errore modello semplice',distances.mean())



#RQ2: Is the fitting of the model improving using a flexibility of alpha (infetivity rate) sensible to comment's toxicity?
    
# Preparo i dati e fitto il modello 
observed_data = np.array([np.datetime64(x.replace(tzinfo=None)).astype(np.int64) for x in dataset_user['created_at']])
mean_lag=np.mean(np.diff(observed_data))
ℋ_t=observed_data-min(observed_data)+mean_lag
𝒯_T=list(dataset_user.toxicity_score)
𝛉_exp=exp_mle_toxicity(ℋ_t,𝒯_T,max(ℋ_t))

# Faccio il QQplot del modello
tsShifted = exp_hawkes_compensators_toxicity(ℋ_t,𝒯_T, 𝛉_exp)
iat = np.diff(np.insert(tsShifted, 0, 0))
qqplot(iat, dist=stats.expon, fit=True, line="45")
plt.show()

# Estraggo una metric per valutare il modello
slope, intercept = sm.OLS(iat, sm.add_constant(stats.expon.ppf((np.arange(1, len(iat) + 1) - 0.5) / len(iat)))).fit().params
print("Pendenza della linea di riferimento adattata:", slope)

slope, intercept = 1, 0
predicted_values = slope * np.arange(1, len(iat) + 1) + intercept

# Calcolo delle distanze
distances = np.abs(iat - predicted_values)

print('errore modello tossicità',distances.mean())





