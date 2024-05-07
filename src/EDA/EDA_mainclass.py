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


# 2.7 - Deep vs Flash conversaion comparison
df=data[data['root_submission'].isin(ids_flash_conversations) | data['root_submission'].isin(ids_deep_chat)]

import matplotlib.pyplot as plt

x = df.groupby('root_submission')['thread_lifetime_h'].max()
y = df.groupby('root_submission')['user'].nunique()
z = df.groupby('root_submission')['number_of_comments'].max()

colors = [
          'red' if root_submission in ids_flash_conversations else
          'blue' if root_submission in ids_deep_chat else
          'gray' for root_submission in x.index]
plt.figure(figsize=(10, 8))
plt.scatter(x, y, s=z*1, c=colors, alpha=0.1)

# Personalizza il plot
plt.xlabel('Durata massima del thread (ore)')
plt.ylabel('Numero unico di utenti')
plt.title('Scatter plot dei thread')
plt.colorbar(label='Numero massimo di commenti')
plt.grid(True)
plt.xlim(0,1000)

plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Appartiene a entrambe le categorie'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Flash Conversations'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Deep Chat'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Non appartiene a nessuna categoria')])

plt.show()

# 2.8 - User Activity distribution 

max_comments_per_user = data.groupby('user')['number_of_comments_by_user_in_thread'].mean()
np.mean(max_comments_per_user<=1)
plt.hist(max_comments_per_user, bins=300, edgecolor='black')
plt.xlabel('90th pe Number of Comments by User in Thread')
plt.ylabel('Frequency')
plt.title('Distribution of 90th Percentile Comments per User in Threads')
plt.axvline(x=2, color='r', linestyle='--')  # Aggiunge una linea per indicare il punto 2 sul grafico
plt.text(30, 3000, '69% of users ', rotation=0, fontsize=12, color='r')
plt.xlim(0,40)
plt.show()

# 2.9 - Definition Deep and Flash, Users and Threads
max_comments_per_user = data.groupby('user')['number_of_comments_by_user_in_thread'].mean()
deep_users = max_comments_per_user[max_comments_per_user > 1.5].index.tolist()
flash_users = max_comments_per_user[max_comments_per_user <=1.5].index.tolist()
data['user_type'] = data['user'].isin(flash_users).apply(lambda x: 'flash' if x else 'deep')

grouped = data.groupby('root_submission')['first_comment'].mean().reset_index()
ids_deep_chat = grouped[grouped.first_comment < 0.4]['root_submission'].unique()
ids_flash_conversations = grouped[grouped.first_comment > 0.8]['root_submission'].unique()


data['thread_type']=data['root_submission'].isin(ids_flash_conversations).apply(lambda x: 'flash' if x else 'deep')
data['thread_type_bool']=data['thread_type']=='deep'

#3.0 - User diet vs User type
df = data.groupby(['user', 'root_submission', 'user_type', 'thread_type'])['comment_id'].count().reset_index()
df.columns
thread_type = df['user_type']=='deep'
user_type =df['thread_type']=='deep'
table = np.array([
    [sum(np.logical_and(thread_type, user_type)), sum(np.logical_and(thread_type, np.logical_not(user_type)))],
    [sum(np.logical_and(np.logical_not(thread_type), user_type)), sum(np.logical_and(np.logical_not(thread_type), np.logical_not(user_type)))]
])

chi2, p, dof, expected = chi2_contingency(table)

print("Chi-square statistic:", chi2)
print("p-value:", p)
print("Degrees of freedom:", dof)
print("Contingency table:")
print(table)
print("Expected frequencies table:")
print(expected)

row_sums = table.sum(axis=1)
relative_table = table / row_sums[:, np.newaxis]
plt.figure(figsize=(8, 6))
sns.heatmap(relative_table, annot=True, cmap='Blues', fmt='.2f')
plt.title('Contingency table User and Thread')
plt.xlabel('Deep User')
plt.ylabel('Deep Thread')
plt.show()












