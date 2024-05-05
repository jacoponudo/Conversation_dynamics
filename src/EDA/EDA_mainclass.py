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
from EDA_package import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

# Create output directory if it doesn't exist
output_dir = '/Users/jacoponudo/Documents/thesis/src/EDA/output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loading the processed data
social_media_name = "voat"
root = '/Users/jacoponudo/Documents/thesis/'
data = pd.read_csv(root + 'src/PRO/output/' + social_media_name + '_processed.csv')


# 2.1 - Participation Distribution by Decile
data['is_toxic']=data['toxicity_score']>0.6
bins = [i / 10 for i in range(11)]
data['Decile'] = pd.cut(data['percentile'] / 100, bins=bins, labels=False, include_lowest=True)
for topic in data['topic'].unique():
    topic_data = data[data['topic'] == topic]
    grouped_data = topic_data.groupby(['root_submission', 'Decile']).agg({'user': ['nunique', 'count']}).reset_index()
    grouped_data.columns = ['root_submission', 'Decile', 'unique_users', 'rows']
    grouped_data['user_activity'] = grouped_data['rows'] / grouped_data['unique_users']
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Decile', y='user_activity', data=grouped_data, hue='Decile', showfliers=False)
    plt.title(f'Distribution of participation per decile of conversation for topic: {topic}')
    plt.xlabel('Decile')
    plt.ylabel('Participation')
    plt.savefig(f'{output_dir}/participation_distribution_{topic}.png')
    plt.close()

data = data.sort_values(by='created_at')
data['first_comment'] = data['sequential_number_of_comment_by_user_in_thread'] == 1
grouped = data.groupby('root_submission')['first_comment'].mean().reset_index()

# 2.2 - Histogram of Share of First Comment per Thread
plt.hist(grouped['first_comment'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Share of first comment')
plt.ylabel('Frequency')
plt.title('Histogram of Share of First Comment per Thread')
plt.axvline(x=0.4, color='blue', linestyle='--', label='deep chat')
plt.axvline(x=0.8, color='yellow', linestyle='--', label='flash conversation')
plt.legend()
plt.savefig(f'{output_dir}/first_comment_share_histogram.png')
plt.close()

# 2.3 - Unique Word Count Analysis
ids_deep_chat = grouped[grouped.first_comment < 0.4]['root_submission'].unique()
ids_flash_conversations = grouped[grouped.first_comment > 0.8]['root_submission'].unique()
deep_chat = data[data['root_submission'].isin(ids_deep_chat)]
flash_conversations = data[data['root_submission'].isin(ids_flash_conversations)]
plt.figure(figsize=(10, 6))
deep_box = plt.boxplot(deep_chat['unique_words_count'], positions=[1], widths=0.5, patch_artist=True,boxprops=dict(facecolor='blue', color='blue'), showfliers=False)
flash_box = plt.boxplot(flash_conversations['unique_words_count'], positions=[2], widths=0.5, patch_artist=True, boxprops=dict(facecolor='yellow', color='yellow'), showfliers=False)
plt.legend([deep_box['boxes'][0], flash_box['boxes'][0]], ['Deep Chat', 'Flash Conversations'])
plt.title('Number of Unique Words')
plt.xlabel('Group')
plt.ylabel('Number of Unique Words')
plt.xticks([1, 2], ['Deep Chat', 'Flash Conversations'])
plt.grid(True)
plt.savefig(f'{output_dir}/unique_word_count_analysis.png')
plt.close()

# 2.4 - Thread Lifetime Distribution Analysis
agg_data = data.groupby('root_submission')['thread_lifetime_h'].min().reset_index()
deep_chat = agg_data[agg_data['root_submission'].isin(ids_deep_chat)]
flash_conversations = agg_data[agg_data['root_submission'].isin(ids_flash_conversations)]
plt.figure(figsize=(10, 6))
plt.boxplot(deep_chat['thread_lifetime_h'], positions=[1], patch_artist=True, boxprops=dict(facecolor='blue', color='blue'),
            showfliers=False, widths=0.5)
plt.boxplot(flash_conversations['thread_lifetime_h'], positions=[2], patch_artist=True,
            boxprops=dict(facecolor='yellow', color='yellow'), showfliers=False, widths=0.5)
plt.legend([deep_box['boxes'][0], flash_box['boxes'][0]], ['Deep Chat', 'Flash Conversations'])
plt.title('Boxplot of Thread Lifetime')
plt.ylabel('Thread Lifetime (hours)')
plt.xticks([1, 2], ['Deep Chat', 'Flash Conversations'])
plt.grid(True)
plt.savefig(f'{output_dir}/thread_lifetime_distribution.png')
plt.close()

# 2.5 - Toxicity Analysis per Topic and Group
categories = data['topic'].unique()
deep_chat_toxic_means = []
flash_conversations_toxic_means = []
for topic in categories:
    topic_data = data[data['topic'] == topic]
    topic_data['is_toxic'] = topic_data['toxicity_score'] > 0.6
    deep_chat = topic_data[topic_data['root_submission'].isin(ids_deep_chat)]
    flash_conversations = topic_data[topic_data['root_submission'].isin(ids_flash_conversations)]
    deep_chat_toxic_mean = deep_chat['is_toxic'].mean()
    flash_conversations_toxic_mean = flash_conversations['is_toxic'].mean()
    deep_chat_toxic_means.append(deep_chat_toxic_mean)
    flash_conversations_toxic_means.append(flash_conversations_toxic_mean)
bar_width = 0.35
index = np.arange(len(categories))
plt.figure(figsize=(10, 6))
bar1 = plt.bar(index, deep_chat_toxic_means, bar_width, label='Deep Chat', color='blue')
bar2 = plt.bar(index + bar_width, flash_conversations_toxic_means, bar_width, label='Flash Conversations', color='yellow')
plt.xlabel('Category')
plt.ylabel('Average Toxicity')
plt.title('Average Toxicity per Category and Group')
plt.xticks(index + bar_width / 2, categories)
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/toxicity_analysis_per_topic_and_group.png')
plt.close()

# 2.6 - Toxicity Distribution per User
deep_chat = data[data['root_submission'].isin(ids_deep_chat)]
flash_conversations = data[data['root_submission'].isin(ids_flash_conversations)]
utenti_primo_dataset = set(deep_chat['user'])
utenti_secondo_dataset = set(flash_conversations['user'])
utenti_solo_primo_dataset = utenti_primo_dataset - utenti_secondo_dataset
utenti_solo_secondo_dataset = utenti_secondo_dataset - utenti_primo_dataset
media_toxic_per_utente = data.groupby('user')['is_toxic'].mean()
utenti_solo_primo_dataset_lista = list(utenti_solo_primo_dataset)
utenti_solo_secondo_dataset_lista = list(utenti_solo_secondo_dataset)
plt.figure(figsize=(10, 6))
bp = plt.boxplot([media_toxic_per_utente.loc[utenti_solo_primo_dataset_lista],
                  media_toxic_per_utente.loc[utenti_solo_secondo_dataset_lista]], showfliers=False,
                 labels=['Deep Chat Users', 'Flash conversations'], patch_artist=True,
                 medianprops=dict(color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'))
bp['boxes'][0].set(facecolor='blue')
bp['boxes'][1].set(facecolor='yellow')
plt.title('Toxicity Distribution per User')
plt.ylabel('% of Toxic comments')
plt.savefig(f'{output_dir}/toxicity_distribution_per_user.png')
plt.close()

# 2.7 - User Activity distribution 

max_comments_per_user = data.groupby('user')['number_of_comments_by_user_in_thread'].quantile(0.9)
np.mean(max_comments_per_user<=1)
plt.hist(max_comments_per_user, bins=100, edgecolor='black')
plt.xlabel('90th pe Number of Comments by User in Thread')
plt.ylabel('Frequency')
plt.title('Distribution of 90th Percentile Comments per User in Threads')
plt.axvline(x=2, color='r', linestyle='--')  # Aggiunge una linea per indicare il punto 2 sul grafico
plt.text(30, 3000, '69% of users ', rotation=0, fontsize=12, color='r')
plt.show()

# create the two groups of users: deep and flash 
max_comments_per_user = data.groupby('user')['number_of_comments_by_user_in_thread'].max()
mean_less_than_or_equal_to_1 = np.mean(max_comments_per_user <= 1)
deep_users = max_comments_per_user[max_comments_per_user > 1].index.tolist()
flash_users = max_comments_per_user[max_comments_per_user <=1].index.tolist()
print("Deep Users:", deep_users)
print("Flash Users:", flash_users)
print("Percentage of users with max comments per thread <= 1:", mean_less_than_or_equal_to_1)

# Extract the two dataset for user deep and flash users
deep_users_data = data[data['user'].isin(deep_users)]
flash_users_data = data[data['user'].isin(flash_users)]

# Gli utenti deep e quelli flash, mangiano gli stessi thread? #Hanno stili di scrittura diversi? #hanno lifetime diversi?
deep_users_number_of_thread_per_user = deep_users_data['root_submission'].unique()
flash_users_number_of_thread_per_user= flash_users_data['root_submission'].unique()
flash_users_threads = set(flash_users_number_of_thread_per_user)
deep_users_threads = set(deep_users_number_of_thread_per_user)
len(flash_users_threads.intersection(deep_users_threads))/len(deep_users_threads)
# Partizionando in maniera causuale gli utenti quale sarebbe lo share di dieta che condibvidono i due gruppi 
#il vaore 20% che otteniamo ora è 


# Quanti commenti sono fanno parte di deep chat e quanti fanno parte di flash interactions
np.mean(data['sequential_number_of_comment_by_user_in_thread']==1)
data[data['sequential_number_of_comment_by_user_in_thread']>1]['user'].unique)()
max_comments_per_user = data.groupby('user')['number_of_comments_by_user_in_thread'].max()
np.mean(max_comments_per_user <= 1)
max_comments_per_user = data.groupby('root_submission')['sequential_number_of_comment_by_user_in_thread'].mean()


#3.0

df=data

G = nx.Graph()

# Aggiunta dei nodi
G.add_nodes_from(df['user'].unique())
from tqdm import tqdm
# Iterazione sul dataframe per aggiungere gli archi
for submission in tqdm(df['root_submission'].unique()):
    users_with_submission = df[df['root_submission'] == submission]['user'].tolist()
    if len(users_with_submission) > 1:
        # Se ci sono più di un utente con la stessa root_submission, aggiungi un arco tra di loro
        for i in range(len(users_with_submission)):
            for j in range(i+1, len(users_with_submission)):
                user1 = users_with_submission[i]
                user2 = users_with_submission[j]
                if user1 != user2:  # Evita i self loop
                    G.add_edge(user1, user2, weight=1)  # Aggiungi l'arco con peso 1


# fai pruning 
grado_minimo = 6
G_pruned = prune_graph(G, grado_minimo)
###

blue_subgraph = G.subgraph(deep_users)
red_subgraph = G.subgraph(flash_users)

# Calcolo dei gradi dei nodi
blue_degrees = [degree for node, degree in blue_subgraph.degree()]
red_degrees = [degree for node, degree in red_subgraph.degree()]

# Plot dei boxplot delle distribuzioni dei gradi
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.boxplot(data=blue_degrees, color='blue')
plt.title('Distribuzione dei gradi dei nodi blu')
plt.xlabel('Grado')
plt.ylabel('Numero di nodi')

plt.subplot(1, 2, 2)
sns.boxplot(data=red_degrees, color='red')
plt.title('Distribuzione dei gradi dei nodi rossi')
plt.xlabel('Grado')
plt.ylabel('Numero di nodi')

plt.tight_layout()
plt.show()




import networkx as nx
import community  # Assicurati di aver installato il pacchetto python-louvain per utilizzare questo modulo

# Supponiamo che tu abbia già creato il tuo grafo G

# Trova la partizione dei nodi utilizzando l'algoritmo di Louvain
partition = community.best_partition(G)

# Identifica due comunità
community_1 = []
community_2 = []
for node, comm in partition.items():
    if comm == 0:
        community_1.append(node)
    elif comm == 1:
        community_2.append(node)

# Stampa i nodi nelle due comunità
print("Comunità 1:", community_1)
print("Comunità 2:", community_2)




