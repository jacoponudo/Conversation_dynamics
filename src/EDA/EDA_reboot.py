




# Notes for Jacopo:
'''
This script from the outpu of PRO do a little bit of EDA to undestand better how thread develope
'''

#### Stage 2 - Exploratory Data Analysis ####

import sys
import os
import numpy as np
module_path = '/Users/jacoponudo/Documents/thesis/src/EDA'
sys.path.append(module_path)
from EDA_package.function import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from tqdm import tqdm 
from scipy.stats import chi2_contingency



# Create output directory if it doesn't exist
output_dir = '/Users/jacoponudo/Documents/thesis/src/EDA/output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loading the processed data
social_media_name = "voat"
root = '/Users/jacoponudo/Documents/thesis/'
input_filename = os.path.join(root, "data", social_media_name, f"{social_media_name}_labeled_data_unified.parquet")
data = pd.read_parquet(input_filename)

# Sample of users 
data=data[data['user'].isin(data.user.sample(100))]



thread_user=data.groupby(['user','root_submission'])['comment_id'].count().reset_index()
thread_user_window=thread_user[thread_user['comment_id']>2]
user_thread_data_sorted = user_thread_data_sorted.sort_values(by='comment_id')

results = []

# Iterate through each comment in the conversation
for i, row in tqdm(thread_user_window.iterrows(), total=len(thread_user)):
    user = row['user']
    root = row['root_submission']
    
    # Filter user_thread_data and sort by 'created_at'
    user_thread_data = data[(data['user'] == user) & (data['root_submission'] == root)]
    user_thread_data_sorted = user_thread_data.sort_values(by='created_at')
    
    user_thread_data_sorted['created_at'] = pd.to_datetime(user_thread_data_sorted['created_at'])
    user_thread_data_sorted['temporal_distance_from_previous_comment_s'] = user_thread_data_sorted['created_at'].diff().dt.total_seconds()
    user_thread_data_sorted['temporal_distance_from_previous_comment_s'] = user_thread_data_sorted['temporal_distance_from_previous_comment_s'].fillna(0)
    
    
    # Calculate temporal distances and toxicity scores dynamically based on comment_id
    temporal_distances = {}
    toxicity_scores = {}
    for j in range(1, len(user_thread_data_sorted) ):
        temporal_distances[f'IAT_{j}'] = (user_thread_data_sorted['temporal_distance_from_previous_comment_s'].iloc[j]) * 60 * 60
        toxicity_scores[f'toxicity_{j}'] = user_thread_data_sorted['toxicity_score'].iloc[j]

    # Append the results to the list
    results.append({'user': user, 'root': root, **temporal_distances, **toxicity_scores})

# Create the final DataFrame
result_df = pd.DataFrame(results)



result_df.to_csv('/Users/jacoponudo/Documents/thesis/src/EDA/output/IAT_Toxicity_by_position.csv')



# PLOT 1
import seaborn as sns
import matplotlib.pyplot as plt

columns_to_plot = ['t2', 't3', 't4', 't5', 't6','t7', 't8', 't9']
columns_to_plot=['toxicity_t2', 'toxicity_t3', 'toxicity_t4', 'toxicity_t5','toxicity_t6', 'toxicity_t7', 'toxicity_t8', 'toxicity_t9']
result_df.columns

melted_df = result_df[columns_to_plot].melt(var_name='Columns', value_name='Values')

plt.figure(figsize=(10, 6))
sns.boxplot(x='Columns', y='Values', data=melted_df,showfliers=False)

plt.title('Boxplot of Inter Arrival Time for Conversations of len9')

plt.show()


# PLOT 2
import matplotlib.pyplot as plt

x_column = 0
y_column = 'toxicity_t2'

plt.figure(figsize=(8, 6))
plt.scatter(standardized_df.iloc[:, x_column], result_df.iloc[:, 10], alpha=0.5, s=2)

plt.title(f'Scatter plot tra {x_column} e {y_column}')
plt.xlabel(x_column)
plt.ylabel(y_column)

plt.show()


