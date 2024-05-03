# Notes for Jacopo:
'''
This script loads labeled data from a Parquet file, filters threads based on the number of comments, and calculates percentiles for comment creation times within each thread.
'''
import sys
module_path = '/Users/jacoponudo/Documents/thesis/src/PRO'
sys.path.append(module_path)
from PRO_package.functions import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os

social_media_name = "voat"
thread_identifier = "root_submission"
root = '/Users/jacoponudo/Documents/thesis/'

input_filename = os.path.join(root, "data", social_media_name, f"{social_media_name}_labeled_data_unified.parquet")
data = pd.read_parquet(input_filename)

#### Stage 1 - Add variables ####
'''
Essential to add some useful variables before to start working. This is the variables list:
    
* thread_lifetime 
* thread_birth
* temporal_distance_birth
* number_of_comments
* number_of_users
* number_of_comments_user_in_thread
* sequence_number_comment_user_thread
'''

# Filter mantaining just threads with more than 50 comments
comment_counts = data[thread_identifier].value_counts()
threads_with_more_than_10_comments = comment_counts[comment_counts > 50].index
data = data[data[thread_identifier].isin(threads_with_more_than_10_comments)]

threads = data.groupby(thread_identifier)  # Divide in threads

progress_bar = tqdm(total=len(threads), desc="Add variables...")
for name, group in threads:
    group['time'] = pd.to_datetime(group['created_at'])
    group.sort_values(by='created_at', inplace=True)
    group['thread_birth'] = group['time'].min()
    group['temporal_distance_birth_h'] = round(pd.to_timedelta(group['time']-group['thread_birth']).dt.total_seconds() / 3600, 1)
    group['thread_lifetime_h'] = round((group['time'].max()-group['time'].min()).total_seconds() / 3600, 1)
    group['number_of_comments'] = len(group)
    group['number_of_users'] = len(group.user.unique())
    group['unique_words_count'] = group['text'].apply(count_unique_words)
    
    percentiles = add_percentile_column(group)['percentile']
    
    user_counts = group.groupby('user')['user'].count()
    group['number_of_comments_by_user_in_thread'] = group['user'].map(user_counts)
    group['user_appearances'] = group.groupby('user').cumcount() + 1
    
    data.loc[group.index, 'percentile'] = percentiles
    data.loc[group.index,'sequential_number_of_comment_by_user_in_thread'] = group['user_appearances']
    data.loc[group.index, 'number_of_comments_by_user_in_thread'] = group['number_of_comments_by_user_in_thread']
    data.loc[group.index, 'thread_birth'] = group['thread_birth']
    data.loc[group.index,'temporal_distance_birth_h'] = group['temporal_distance_birth_h']
    data.loc[group.index, 'thread_lifetime_h'] = group['thread_lifetime_h']
    data.loc[group.index, 'number_of_users'] = group['number_of_users']  
    data.loc[group.index, 'unique_words_count'] = group['unique_words_count']
    data.loc[group.index, 'number_of_comments'] = group['number_of_comments']
    progress_bar.update(1)
    
progress_bar.close()


data.to_csv(root+'src/PRO/output/'+social_media_name+'_processed.csv')




