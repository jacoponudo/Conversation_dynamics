# Fitting Parameters 

import powerlaw
import warnings
from scipy.stats import beta
import pandas as pd
import numpy as np

def estimate_parameters_U(fb,xmin_fb=50):
  users_fb = fb.groupby('post_id')['user_id'].nunique().reset_index()
  users_fb.columns = ['post_id', 'user_count']
  users_fb = users_fb[users_fb['user_count'] > xmin_fb]
  observed_fb = users_fb['user_count']
  gamma_fb = fit_power_law(observed_fb, xmin_fb)
  return gamma_fb

def fit_power_law(data, xmin):
    fit = powerlaw.Fit(data, xmin=xmin, discrete=True)
    return fit.alpha
def fit_beta_distribution(data):
    data = data[data > 0]
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='The iteration is not making good progress')
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in sqrt')

    a, b, loc, scale = beta.fit(data)
    return {'a': a, 'b': b, 'loc': loc, 'scale': scale}
def simulate_number_of_comments(alpha, lambda_,):
    # Simula la componente inflazionata (produce 0 con probabilità alpha)
    inflate = np.random.binomial(1, alpha, 1)
    # Simula la componente contatore (distribuzione esponenziale negativa)
    counts = np.random.exponential(1/lambda_, 1)
    # Discretizza i valori esponenziali per ottenere valori di conteggio interi
    counts = np.round(counts).astype(int)
    counts[counts<0]=0
    # Combina le componenti inflazionate e di conteggio
    simulated_data = inflate * (counts)
    return simulated_data
def simulate_zip(alpha, lambda_, size=10000):
    # Simula la componente inflazionata (produce 0 con probabilità alpha)
    inflate = np.random.binomial(1, alpha, size)
    # Simula la componente contatore (distribuzione esponenziale negativa)
    counts = np.random.exponential(1/lambda_, size)
    # Discretizza i valori esponenziali per ottenere valori di conteggio interi
    counts = np.round(counts).astype(int)
    counts[counts<0]=0
    # Combina le componenti inflazionate e di conteggio
    simulated_data = inflate * counts
    return simulated_data

# Function to calculate KL-divergence
def kl_divergence(p, q):
    epsilon = 1e-10  # Small constant to avoid log(0)
    return np.sum(p * np.log((p + epsilon) / (q + epsilon)))

def process_platform(df, platform_name):
  df['last_comment']=df['sequential_number_of_comment_by_user_in_thread']==df['number_of_comments_by_user_in_thread'].copy()
  df=df.dropna(subset='IAT_user_thread').copy()
  df['time_difference']=df['IAT_user_thread']/(60*60*100)
  merged_df=df[df['last_comment']==False].copy()
  merged_df_final=df[df['last_comment']==True].copy()
  return merged_df,merged_df_final


import numpy as np
from tqdm import tqdm
from scipy.stats import burr
from scipy.special import kl_div

def process_social_platform(names, datas):
    params_dict = {}

    for i, df in tqdm(enumerate(datas), total=len(names)):
        social = names[i]
        params_dict[social] = {}

        # Estimate gamma parameter
        params_dict[social]['gamma'] = estimate_parameters_U(df)

        # Fit beta distribution parameters
        time = df[df['sequential_number_of_comment_by_user_in_thread'] == 1]['temporal_distance_birth_base_100h']
        time_params = fit_beta_distribution(time)
        params_dict[social]['a'] = time_params['a']
        params_dict[social]['b'] = time_params['b']
        params_dict[social]['loc'] = time_params['loc']
        params_dict[social]['scale'] = time_params['scale']

        # Find best alpha and lambda for Zipf-like distribution
        size_of_interaction = df.groupby(['user_id', 'post_id'])['comment_id'].count().reset_index()
        observed_data = size_of_interaction['comment_id'] - 1
        alpha_range = np.arange(0, 1, 0.05)
        lambda_range = np.arange(0.1, 2, 0.05)
        best_alpha = None
        best_lambda = None
        best_loss = np.inf

        for alpha in alpha_range:
            for lambda_ in lambda_range:
                simulated_data = simulate_zip(alpha, lambda_, size=len(observed_data))
                max_value = max(np.max(observed_data), np.max(simulated_data))
                observed_counts = np.bincount(observed_data, minlength=max_value + 1)
                simulated_counts = np.bincount(simulated_data, minlength=max_value + 1)
                observed_distribution = observed_counts / np.sum(observed_counts)
                simulated_distribution = simulated_counts / np.sum(simulated_counts)
                loss = kl_divergence(observed_distribution, simulated_distribution)
                if loss < best_loss:
                    best_loss = loss
                    best_alpha = alpha
                    best_lambda = lambda_

        params_dict[social]['alpha'] = np.round(best_alpha, 3)
        params_dict[social]['lambda'] = best_lambda

        # Process IAT and IAT_f
        IAT, IAT_f = process_platform(df, social)

        # Fit Burr distribution parameters
        data = IAT['time_difference']
        data = data[data > 0]
        params_dict[social]['c'], params_dict[social]['d'], params_dict[social]['l'], params_dict[social]['s'] = burr.fit(data)

        data = IAT_f['time_difference']
        data = data[data > 0]
        params_dict[social]['cf'], params_dict[social]['df'], params_dict[social]['lf'], params_dict[social]['sf'] = burr.fit(data)
        
        # Fit Beta for stimulus
        params_dict[social]['ka'], params_dict[social]['kb'], params_dict[social]['kloc'], params_dict[social]['kscale']=estimate_stimulus_reply(df)

    return params_dict





def estimate_stimulus_reply(social):
    # Convert 'created_at' to datetime
    social['created_at'] = pd.to_datetime(social['created_at'])

    # Sort the data by 'post_id' and 'created_at'
    social = social.sort_values(by=['post_id', 'created_at'])

    # Assign a sequential index to each comment within each 'post_id' group
    social['indice_commento'] = social.groupby('post_id').cumcount()

    # Select columns of interest
    df = social[['indice_commento', 'user_id', 'post_id', 
                 'sequential_number_of_comment_by_user_in_thread', 
                 'number_of_comments_by_user_in_thread', 
                 'number_of_comments']].copy()

    # Calculate the distance between comments
    df['distanza_tra_commenti'] = df.groupby(['user_id', 'post_id'])['indice_commento'].diff()
    df['distanza_tra_commenti_relativa'] = df['indice_commento'] / df['number_of_comments']

    # Filter for specific values
    df_filtered = df[(df['sequential_number_of_comment_by_user_in_thread'] == 3) & 
                     (df['number_of_comments_by_user_in_thread'] != df['sequential_number_of_comment_by_user_in_thread'])]

    # Extract the filtered data
    data_to_fit = df_filtered['distanza_tra_commenti_relativa']

    # Check for finite values
    data_to_fit = data_to_fit[np.isfinite(data_to_fit)]

    # Fit the Beta distribution to the data
    a, b, loc, scale = beta.fit(data_to_fit)

    return a, b, loc, scale