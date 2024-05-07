# Notes for Jacopo:
''' 

'''

# Define the module path + import packages
import sys
module_path = '/Users/jacoponudo/Documents/thesis/src/UAA'
sys.path.append(module_path)
from UAA_package.NLP_tools import *
from UAA_package.functions import *

# Create output directory if it doesn't exist
output_dir = '/Users/jacoponudo/Documents/thesis/src/UAA/output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loading the processed data
social_media_name = "voat"
root = '/Users/jacoponudo/Documents/thesis/'
data = pd.read_csv(root + 'src/PRO/output/' + social_media_name + '_processed.csv')

# Estract informations from columns
#* La lunghezza mediana dei testi( numero di parole )
word_count_per_post = data.groupby('user')['text'].apply(lambda x: x.fillna('').apply(lambda text: len(str(text).split()))).reset_index()
lunghezza=word_count_per_post.groupby('user')['text'].median().reset_index()
lunghezza.columns=['user', 'median_length_comment']

#* Numero unico di parole pronunciate dall’utente (post preprocessing) 
word_count_per_user=data.groupby('user')['unique_word_user'].max().reset_index()

#* La media della tossicità dei commenti 
tox_median_per_user=data.groupby('user')['toxicity_score'].median().reset_index()
tox_median_per_user.columns=['user', 'median_toxicity_score']

#* Il lifetime dell'utente nei dati
results = []
for topic in data['topic'].unique():
    df = data[data['topic'] == topic].copy()
    df['created_at'] = pd.to_datetime(df['created_at'])
    user_comment_dates = df.groupby('user')['created_at'].agg(['min', 'max'])
    user_comment_dates['days_lifetime'] = (user_comment_dates['max'] - user_comment_dates['min']).dt.days
    user_comment_dates.reset_index(inplace=True)
    user_comment_dates['topic'] = topic
    user_comment_dates=user_comment_dates[['user','days_lifetime','topic']]
    results.append(user_comment_dates)
final_df = pd.concat(results, ignore_index=True)
final_df['days_lifetime_topic'] = final_df['days_lifetime'].astype(str) + ' ' + final_df['topic']
final_pivot = final_df.pivot_table(index='user', columns='topic', values='days_lifetime', aggfunc='first')
final_pivot.reset_index(inplace=True)
final_pivot.fillna(0, inplace=True)
final_pivot.columns = ['user'] + [f'days_lifetime_{topic}' for topic in final_pivot.columns[1:]]

#* Il numero medio di commenti per thread. 
mean_number_of_comment_per_thread=data.groupby(['user','root_submission'])['comment_id'].nunique().reset_index().groupby('user')['comment_id'].mean().reset_index()
mean_number_of_comment_per_thread.columns=['user','mean_number_of_comments_thread']

#* Numero di commenti
number_of_comment_per_user=data.groupby(['user'])['comment_id'].nunique().reset_index()
number_of_comment_per_user.columns=['user','number_of_comments']


# Put all the columns togheter
merged_data = lunghezza.merge(word_count_per_user, on='user')
merged_data = merged_data.merge(tox_median_per_user, on='user')
merged_data = merged_data.merge(final_pivot, on='user')
merged_data = merged_data.merge(mean_number_of_comment_per_thread, on='user')
merged_data = merged_data.merge(number_of_comment_per_user,on='user')

# Plot correlation matrix among variables
correlation_matrix = merged_data[['median_length_comment', 'unique_word_user',
       'median_toxicity_score', 'days_lifetime_Conspiracy',
       'days_lifetime_News', 'days_lifetime_Politics','mean_number_of_comments_thread','number_of_comments']].corr()

# Plot della matrice di correlazione utilizzando seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()





df=merged_data[merged_data['number_of_comments']>30]
colors = df['median_toxicity_score']
labels = df['user']

plt.figure(figsize=(10, 8))
plt.scatter( df['median_toxicity_score'],df['unique_word_user'], c=colors, cmap='coolwarm', alpha=0.7, label=labels)
plt.title('Scatter Plot: Unique Word Count vs Median Toxicity Score')
plt.ylabel('Unique Word Count per User')
plt.xlabel('Median Toxicity Score')
plt.grid(True)


plt.show()





