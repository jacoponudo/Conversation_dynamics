import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta, burr
import powerlaw
import random
import math


def simulate_number_of_users(gamma_fb, min_users, size=1):
    # Simulate the number of users using a power-law distribution
    return min_users * (1 - np.random.rand(size))**(-1 / (gamma_fb - 1))

def simulate_initial_comment(a, b, loc, scale, size=1):
    # Simulate the initial comment using a Beta distribution
    return beta.rvs(a, b, loc, scale, size)

def simulate_number_of_comments(alpha, lambda_, size=1):
    # Simulate the inflated component (produces 0 with probability alpha)
    inflate = np.random.binomial(1, alpha, size)
    # Simulate the count component (negative exponential distribution)
    counts = np.random.exponential(1 / lambda_, size)
    # Discretize the exponential values to obtain integer count values
    counts = np.round(counts).astype(int)
    counts[counts < 0] = 0
    # Combine the inflated and count components
    simulated_data = inflate * counts
    return simulated_data



def IAT(c, d, l, s,T=1):
  x = 10
  while x > T:
      x = burr.rvs(c, d, l, s, size=1)
  return x


def simulate_data(social, parameters, num_threads=False, activate_tqdm=True, min_users=50):
    gamma=parameters['gamma']
    a=parameters['a']
    b=parameters['b']
    loc=parameters['loc']
    scale=parameters['scale']
    alpha=parameters['alpha']
    lambda_=parameters['lambda']
    c=parameters['c']
    d=parameters['d']
    l=parameters['l']
    s=parameters['s']
    cf=parameters['cf']
    d_f=parameters['df']
    lf=parameters['lf']
    sf=parameters['sf']
    ka=parameters['ka']
    kb=parameters['kb']
    kloc=parameters['kloc']
    kscale=parameters['kscale']
    data = []
    if num_threads:
        num_threads = min(num_threads, len(social['post_id'].unique()))
        thread_ids = random.sample(list(social['post_id'].unique()), num_threads)
    else:
        thread_ids = social['post_id'].unique()
    
    if activate_tqdm:
        thread_ids = tqdm(thread_ids)
    
    for th in thread_ids:
        thread = social[social['post_id'] == th]
        number_of_users = int(np.round(simulate_number_of_users(gamma, min_users, size=1)[0]))
        T0s = simulate_initial_comment(a, b, loc, scale, size=number_of_users)

        for i in range(number_of_users):
            N = int(simulate_number_of_comments(alpha, lambda_,1)[0] + 1)
            additional_timings=[]
            final_comment_additional_timings=[]
            T=1-T0s[i]
            if N > 1:
              for j in range(N-1):
                if j<(N-2):
                  lag = IAT(c=c, d=d, l=l, s=s,T=T)[0]
                  additional_timings.append(lag)
                  T=1-(T0s[i]+np.sum(additional_timings))
                else:
                  lag=IAT(c=cf, d=d_f, l=lf, s=sf,T=T)[0]
                  final_comment_additional_timings.append(lag)
              timing = np.concatenate([[T0s[i]], additional_timings, final_comment_additional_timings])
            else:
                timing = np.array([T0s[i]])
            timing = np.cumsum(timing)
            timing = [1 if x > 1 else x for x in timing]

            for j, t in enumerate(timing):
                data.append({'user_id': f'User_{i}', 'post_id': th, 'temporal_distance_birth_base_100h': t, 'sequential_number_of_comment_by_user_in_thread': j + 1})

    simulated = pd.DataFrame(data)
    observed = social[social['post_id'].isin(simulated['post_id'].unique())][['user_id', 'post_id', 'temporal_distance_birth_base_100h', 'sequential_number_of_comment_by_user_in_thread']]

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

def plot_ECDF(df, level=95):
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='Time Grid Value', y='Share', hue='Platform', err_style='band', errorbar=('ci', level))
    sns.lineplot(data=df, x='Time Grid Value', y='Share_cc', hue='Platform', err_style='band', errorbar=('ci', level), linestyle='dotted') 
    plt.title('Distribution of Conversation Lifetime Across Percentiles (Reddit vs Facebook)')
    plt.ylabel('Fraction of comments')
    plt.xlabel('Time (base 100)')
    plt.grid(False)
    plt.legend(title='Platform')
    plt.show()

def calculate_loss(observed, simulated):
    combined_results = pd.merge(observed, simulated, on=['post_id', 'Time Grid Value'], suffixes=('_Observed', '_Simulated'))
    combined_results['Errors'] = abs(combined_results['Share_Simulated'] - combined_results['Share_Observed']) + abs(combined_results['Share_cc_Simulated'] - combined_results['Share_cc_Observed'])
    total_error = combined_results['Errors'].sum()

    return total_error


def fit_power_law(data, xmin):
    fit = powerlaw.Fit(data, xmin=xmin, discrete=True)
    return fit.alpha


def fit_beta_distribution(data):
    data = data[data > 0]  # Filter out zero or negative values
    a, b, loc, scale = beta.fit(data)  # , floc=0, fscale=1)
    return {'a': a, 'b': b, 'loc': loc, 'scale': scale}

def simulate_data_M2(social, parameters, num_threads=False, activate_tqdm=True, min_users=50):
    gamma=parameters['gamma']
    a=parameters['a']
    b=parameters['b']
    loc=parameters['loc']
    scale=parameters['scale']
    alpha=parameters['alpha']
    lambda_=parameters['lambda']
    c=parameters['c']
    d=parameters['d']
    l=parameters['l']
    s=parameters['s']
    cf=parameters['cf']
    df=parameters['df']
    lf=parameters['lf']
    sf=parameters['sf']
    ka=parameters['ka']
    kb=parameters['kb']
    kloc=parameters['kloc']
    kscale=parameters['kscale']
    
    data = []
    if num_threads:
        num_threads = min(num_threads, len(social['post_id'].unique()))
        thread_ids = random.sample(list(social['post_id'].unique()), num_threads)
    else:
        thread_ids = social['post_id'].unique()
    
    if activate_tqdm:
        thread_ids = tqdm(thread_ids)

    for th in thread_ids:
        thread = social[social['post_id'] == th]
        number_of_users = int(np.round(simulate_number_of_users(gamma, min_users, size=1)[0]))
        T0s = simulate_initial_comment(a, b, loc, scale, size=number_of_users)
        Ns = simulate_number_of_comments(alpha, lambda_, number_of_users) + 1
        thread = [[T0s[i]] + [np.nan] * (Ns[i] - 1) for i in range(number_of_users)]
        thread = positioning_replies(thread,c, d, l, s,cf, df, lf, sf)
        
        for u, interaction in enumerate(thread):
            timing = np.cumsum(np.array(interaction, dtype=float))
            timing = [x for x in timing if x <= 1]
            for j, t in enumerate(timing):
                data.append({'user_id': f'User_{u}', 'post_id': th, 'temporal_distance_birth_base_100h': t, 'sequential_number_of_comment_by_user_in_thread': j + 1})

    simulated = pd.DataFrame(data)
    observed = social[social['post_id'].isin(simulated['post_id'].unique())][['user_id', 'post_id', 'temporal_distance_birth_base_100h', 'sequential_number_of_comment_by_user_in_thread']]

    return simulated, observed

def positioning_replies(thread,c, d, l, s,cf, df, lf, sf):
    while any(np.isnan(value) for sublist in thread for value in sublist):
        for i,interaction in enumerate(thread):
            j=first_nan(interaction)
            if j!=-1:
                lag=burr.rvs(c, d, l, s, size=1)[0]
                lag_f=burr.rvs(cf, df, lf, sf, size=1)[0]
                if j<(len(interaction)-1):
                    thread[i][j]=float( thread[i][j-1]+lag )
                else:
                    thread[i][j]=float( thread[i][j-1]+ lag_f)
            
    thread = [[min(1, value) for value in sublist] for sublist in thread]
    return thread


def generate_power_law_samples(gamma, lower_bound, sample_size):
    """
    Generate samples from a power-law distribution.

    Parameters:
    gamma (float): The scaling exponent of the power-law distribution.
    lower_bound (float): The lower bound of the distribution.
    sample_size (int): The number of samples to generate.

    Returns:
    np.ndarray: Array of generated samples.
    """
    # The shape parameter 'a' for scipy's powerlaw distribution
    a = gamma - 1

    # Generate uniform random samples in the range (0, 1)
    uniform_samples = np.random.uniform(0, 1, sample_size)

    # Use the inverse transform sampling method to generate power-law samples
    power_law_samples = lower_bound * (1 - uniform_samples) ** (-1 / a)

    return power_law_samples

# Funzione per creare istogrammi di confronto
def plot_histogram_comparison(original, sampled, platform_name, bins):
    plt.figure(figsize=(12, 8))
    sns.histplot(original, kde=True, color='blue', label='Original', bins=bins, stat='density', alpha=0.6)
    sns.histplot(sampled, kde=True, color='red', label='Sampled', bins=bins, stat='density', alpha=0.6)
    plt.title(f'Histogram Comparison for {platform_name}')
    plt.xlabel('Number of Unique Users')
    plt.ylabel('Density')
    plt.legend()
    plt.xlim(0,max(bins))
    plt.show()

# Funzione per calcolare la perdita come somma delle differenze di frequenza bin per bin
def calculate_frequency_difference_loss(original, sampled, bins):
    # Handling NaN values in original and sampled data

    # Compute histograms
    sampled_hist, _ = np.histogram(sampled, bins=bins, density=True)
    original_hist, _ = np.histogram(original, bins=bins, density=True)

    # Calculate loss
    loss = np.sum(np.abs(original_hist - sampled_hist))

    return loss

def simulate_number_of_comments(alpha, lambda_,size):
    # Simula la componente inflazionata (produce 0 con probabilità alpha)
    inflate = np.random.binomial(1, alpha, size)
    # Simula la componente contatore (distribuzione esponenziale negativa)
    counts = np.random.exponential(1/lambda_, size)
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


def last_value_not_na(lista):
    for valore in reversed(lista):
        if not math.isnan(valore):
            return valore
    return None

def first_nan(lista):
    for i, valore in enumerate(lista):
        if math.isnan(valore):
            return i
    return -1  # Se non c'è nessun NaN nella lista

