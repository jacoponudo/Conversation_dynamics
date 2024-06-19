import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def simulate_inital_comment(alpha, beta, size=1):
    return np.random.beta(alpha, beta, size)

def simulate_number_of_comments(alpha, lambd):
    if np.random.rand() < alpha:
        return 0
    else:
        return np.random.poisson(lambd)

def simulate_data(social, alpha, lambda_, mu, sd, a, b, k=1.0, num_threads=100):
    data = []
    for th in tqdm(social['post_id'].unique()[:num_threads]):
        thread = social[social['post_id'] == th]
        number_of_users = thread['user_id'].nunique()
        T0s = simulate_inital_comment(a, b, size=number_of_users)

        for i in range(number_of_users):
            T0 = T0s[i]
            N = int(simulate_number_of_comments(alpha, lambda_) + 1)
            timing = [0] * N

            for j in range(N):
                if j == 0:
                    timing[j] = T0
                elif j == N-1:
                    timing[j] = abs(np.random.normal(mu, sd) )
                else:
                    timing[j] = abs(np.random.normal(mu * k, sd) )

            timing = np.cumsum(timing)
            timing = [x for x in timing if x <= 1]

            for t in timing:
                data.append({'user_id': f'User_{i}', 'post_id': th, 'temporal_distance_birth_base_1000h': t})

    simulated = pd.DataFrame(data)
    observed = social[social['post_id'].isin(simulated['post_id'].unique())][['user_id', 'post_id', 'temporal_distance_birth_base_1000h']]

    return simulated, observed

def calculate_ECDF(df, time_intervals):
    results_list = []
    grouped = df.groupby('post_id')['temporal_distance_birth_base_1000h']

    for post_id, group_data in tqdm(grouped, desc=f"Processing DataFrame"):
        results = pd.DataFrame(index=time_intervals)
        total_comments = len(group_data)

        for time in time_intervals:
            comments_within_time = np.sum(group_data < time)
            share = comments_within_time / total_comments
            results.at[time, post_id] = share
        results = results.stack().reset_index()
        results.columns = ['Time Grid Value', 'post_id', 'Share']
        results_list.append(results)
    final_results = pd.concat(results_list, ignore_index=True)

    return final_results

def plot_ECDF(df,level=95):
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='Time Grid Value', y='Share', hue='Platform', err_style='band', ci=level)
    plt.title('Distribution of Conversation Lifetime Across Percentiles (Reddit vs Facebook)')
    plt.ylabel('Lifetime (minutes)')
    plt.xlabel('Percentile')
    plt.grid(False)
    plt.legend(title='Platform')
    plt.show()
    
def calculate_loss(observed, simulated):
    combined_results = pd.merge(observed, simulated, on=['post_id', 'Time Grid Value'], suffixes=('_Observed', '_Simulated'))
    combined_results['Errors'] = abs(combined_results['Share_Simulated'] - combined_results['Share_Observed'])
    total_error = combined_results['Errors'].sum()

    return total_error