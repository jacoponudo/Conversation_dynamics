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
from scipy import stats
import random 
from scipy.stats import chi2

# Set source
source_data='/Users/jacoponudo/Downloads/voat_labeled_data_unified.parquet'
root='/Users/jacoponudo/Documents/thesis/'
output=root+'src/HWK/output'
output_threads=root+'src/HWK/output/temporary'

# Read the dataset
dataset = pd.read_parquet(source_data)

# Set up parameters of the analysis 
STEP_1=True
STEP_2=False
STEP_3=False
n_threads = 30
thread =0

if STEP_1: 
    print('...starting RUN_THREADS...')
    # Select active users
    users=select_users_with_multiple_comments(dataset, min_comments_per_post=2, min_post_count=1)
    
    # Calculate the number of users per thread
    users_per_thread = len(users) // n_threads
    
    # Calculate the starting and ending indices for the selected thread
    # If it's the last thread, assign the remaining users to it
    start_index = thread * users_per_thread
    end_index = (thread + 1) * users_per_thread
    if thread == n_threads - 1:
        end_index = len(users)
    
    # Select users for the specified thread
    selected_users = users[start_index:end_index]
    user_analysis=analyze_users(dataset, selected_users, 2)
    user_analysis=pd.DataFrame(user_analysis)
    user_analysis.to_csv(output_threads+'/user_analysis_'+str(thread)+'_'+str(n_threads)+'.csv')

if STEP_2: 
    print('...starting MERGE_THREADS...')
    # Get all files in the folder
    all_files = os.listdir(output_threads)
    
    # Filter files that end with n_threads
    filtered_files = [file for file in all_files if file.endswith(str(n_threads)+".csv")]
    
    # Read and merge all filtered files
    dfs = []
    for file in filtered_files:
        file_path = os.path.join(output_threads, file)
        df = pd.read_csv(file_path)
        dfs.append(df)
    
    # Merge all DataFrames together
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df
    merged_df.to_csv(output+'/user_analysis_'+str(users_per_thread)+'.csv',index=False)
    
if STEP_3: 
    print('...starting PLOT_ANALYSIS...')
    # Grafico
    import matplotlib.pyplot as plt
    
    # Definisci i colori in base alla size di alpha
    colors = user_analysis['number_of_comments']
    
    # Crea lo scatterplot 
    plt.figure(figsize=(10, 6))
    plt.scatter(user_analysis['AIC'], user_analysis['AIC_T'], c=colors, cmap='viridis', alpha=0.7)
    
    # Imposta gli stessi limiti sugli assi x e y
    max_lim = max(user_analysis['AIC'].max(), user_analysis['AIC_T'].max())
    min_lim = min(user_analysis['AIC'].min(), user_analysis['AIC_T'].min())
    plt.xlim(min_lim-100, max_lim+100)
    plt.ylim(min_lim-100, max_lim+100)
    
    # Aggiungi etichette agli assi e un titolo
    plt.xlabel('AIC')
    plt.ylabel('AIC_T')
    plt.title('Scatterplot di AIC e AIC_T')
    
    # Aggiungi una legenda per i colori
    plt.colorbar(label='Number of Comments')
    
    # Aggiungi una linea diagonale
    plt.plot([min_lim, max_lim], [min_lim, max_lim], color='red', linestyle='--')
    
    # Mostra il plot
    plt.show()
