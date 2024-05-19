# Notes for Jacopo:
'''
This EDAA  is for Walter, in order to find togheter a mattern or a signal that needs to be investigated
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

output_dir = '/Users/jacoponudo/Documents/thesis/src/EDA/output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loading data, sampling data
social_media_name = "voat"
root = '/Users/jacoponudo/Documents/thesis/'
input_filename = os.path.join(root, "data", social_media_name, f"{social_media_name}_labeled_data_unified.parquet")
data = pd.read_parquet(input_filename)
data=data[data['user'].isin(data.user.sample(1000))]


# Create a dataset that at user x thread level, pairs Inter Arrival Time & Position.
thread_user=data.groupby(['user','root_submission'])['comment_id'].count().reset_index()
thread_user_window=thread_user[thread_user['comment_id']>5]
thread_user_window_sorted = thread_user_window.sort_values(by='comment_id')
results = []

for i, row in tqdm(thread_user_window_sorted.iterrows(), total=len(thread_user_window_sorted)):
    user = row['user']
    root = row['root_submission']
    
    user_thread_data = data[(data['user'] == user) & (data['root_submission'] == root)]
    user_thread_data_sorted = user_thread_data.sort_values(by='created_at')
    
    user_thread_data_sorted['created_at'] = pd.to_datetime(user_thread_data_sorted['created_at'])
    user_thread_data_sorted['temporal_distance_from_previous_comment_s'] = user_thread_data_sorted['created_at'].diff().dt.total_seconds()
    user_thread_data_sorted['temporal_distance_from_previous_comment_s'] = user_thread_data_sorted['temporal_distance_from_previous_comment_s'].fillna(0)
    
    temporal_distances = {}
    toxicity_scores = {}
    number_of_comments= len(user_thread_data_sorted)
    for j in range(1, number_of_comments):
        temporal_distances[f'IAT_{j}'] = (user_thread_data_sorted['temporal_distance_from_previous_comment_s'].iloc[j]) 
        toxicity_scores[f'toxicity_{j}'] = user_thread_data_sorted['toxicity_score'].iloc[j]

    results.append({'user': user, 'root': int(root),'number_of_comments':number_of_comments, **temporal_distances, **toxicity_scores})

result_df = pd.DataFrame(results)

result_df.to_csv('/Users/jacoponudo/Documents/thesis/src/EDA/output/IAT_Toxicity_by_position.csv')

result_df.columns()





# 1 Il valore  maggiore di IAT dove si posiziona?

max_positions=[]

for i,row in result_df.iterrows():
    positions=[]
    n=row['number_of_comments']
    for i in range(n-1):
        positions.append((i+1)/n)
    IAT=list(row[3:2+n])
    max_positions.append(positions[IAT.index(max(IAT))])
    
    

    



# L'inter arrival time tende a essere piu lungo alla fine e all inizio 

bins = np.arange(0, 1.01, 0.2)

# Creazione dell'istogramma
plt.hist(max_positions, bins=bins, edgecolor='black', align='left')

# Aggiunta del titolo e delle etichette degli assi
plt.title('Istogramma delle Posizioni Massime')
plt.xlabel('Posizione del commento')
plt.ylabel('Frequenza')

# Mostra l'istogramma
plt.show()






# Se non c'è un effetto a parabola dell'IAT, ci sono delle alterazioni della  tossicità?










'''

# PLOT 1
import seaborn as sns
import matplotlib.pyplot as plt

columns_to_plot = ['t2', 't3', 't4', 't5', 't6','t7', 't8', 't9']

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

'''
