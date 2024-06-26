import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta
from scipy.stats import burr
import powerlaw

# Supponendo che queste variabili siano già definite: alpha, lambda_, c, d, l, s, T0s


def simulate_number_of_users(gamma_fb, min_users, size=1):
    return (xmin * (1 - np.random.rand(size))**(-1 / (alpha - 1)))

def simulate_inital_comment(a, b,loc,scale, size=1):
    return  beta.rvs(a, b, loc, scale, size)
    
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
    


def simulate_data(social,gamma, a, b,loc,scale, alpha, lambda_,c,d,l,s,cf, df, lf, sf, num_threads=100, activate_tqdm=True,min_users=50):
    data = []
    thread_ids = social['post_id'].unique()[:num_threads]
    
    # Use tqdm conditionally
    if activate_tqdm:
        thread_ids = tqdm(thread_ids)
    
    for th in thread_ids:
        thread = social[social['post_id'] == th]
        number_of_users = simulate_number_of_users(gamma, min_users, size=1)
        T0s = simulate_inital_comment( a, b,loc,scale, size=number_of_users)

        for i in range(number_of_users):
            T0 = T0s[i]
            N = int(simulate_number_of_comments(alpha, lambda_) + 1)
            if N > 1:
                additional_timings = burr.rvs(c, d, l, s, size=N-2)
                final_comment_additional_timings=burr.rvs(cf, df, lf, sf, size=1)
                timing = np.concatenate(([T0], additional_timings,final_comment_additional_timings))
            else:
                timing = np.array([T0])
            timing = timing.tolist()
            timing = np.cumsum(timing)
            timing = [x for x in timing if x <= 1]

            for i,t in enumerate(timing):
                data.append({'user_id': f'User_{i}', 'post_id': th, 'temporal_distance_birth_base_100h': t,'sequential_number_of_comment_by_user_in_thread': i+1})

    simulated = pd.DataFrame(data)
    observed = social[social['post_id'].isin(simulated['post_id'].unique())][['user_id', 'post_id', 'temporal_distance_birth_base_100h','sequential_number_of_comment_by_user_in_thread']]

    return simulated, observed

def calculate_ECDF(df, time_intervals, activate_tqdm=True):
    results_list = []
    df_c = df[df['sequential_number_of_comment_by_user_in_thread'] != 1]
    grouped = df.groupby('post_id')[['temporal_distance_birth_base_100h', 'sequential_number_of_comment_by_user_in_thread']]
    
    if activate_tqdm:
        grouped = tqdm(grouped, desc="Processing DataFrame")
    
    for post_id, group_data in grouped:
        results = pd.DataFrame(index=time_intervals)
        total_comments = len(group_data)
        
        for time in time_intervals:
            comments_within_time = np.sum(group_data['temporal_distance_birth_base_100h'] < time)
            comments_within_time_cc = np.sum(group_data[group_data['sequential_number_of_comment_by_user_in_thread'] != 1]['temporal_distance_birth_base_100h'] < time)
            
            share = comments_within_time / total_comments
            share_cc = comments_within_time_cc / total_comments
            
            results.at[time, 'Share'] = share
            results.at[time, 'Share_cc'] = share_cc
        
        results['post_id'] = post_id
        results = results.reset_index().rename(columns={'index': 'Time Grid Value'})
        results_list.append(results)
    
    final_results = pd.concat(results_list, ignore_index=True)

    return final_results



def plot_ECDF(df,level=95):
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='Time Grid Value', y='Share', hue='Platform', err_style='band',errorbar=('ci', level))
    sns.lineplot(data=df, x='Time Grid Value', y='Share_cc', hue='Platform', err_style='band',errorbar=('ci', level) , linestyle='dotted') 
    plt.title('Distribution of Conversation Lifetime Across Percentiles (Reddit vs Facebook)')
    plt.ylabel('Fraction of comments')
    plt.xlabel('Time (base 100)')
    plt.grid(False)
    plt.legend(title='Platform')
    plt.show() 
    
def calculate_loss(observed, simulated):
    combined_results = pd.merge(observed, simulated, on=['post_id', 'Time Grid Value'], suffixes=('_Observed', '_Simulated'))
    combined_results['Errors'] = abs(combined_results['Share_Simulated'] - combined_results['Share_Observed'])+abs(combined_results['Share_cc_Simulated'] - combined_results['Share_cc_Observed'])
    total_error = combined_results['Errors'].sum()

    return total_error

def fit_power_law(data, xmin):
    fit = powerlaw.Fit(data, xmin=xmin, discrete=True)
    return fit.alpha

def fit_beta_distribution(data):
    data = data[data > 0]  # Filter out zero or negative values
    a, b, loc, scale = beta.fit(data  )# ,floc=0, fscale=1)
    return {'a': a, 'b': b, 'loc': loc, 'scale': scale}

