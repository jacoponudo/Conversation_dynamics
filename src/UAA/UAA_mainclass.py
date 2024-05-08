# Notes for Jacopo:
''' 

'''

# Define the module path + import packages
import sys
module_path = '/Users/jacoponudo/Documents/thesis/src/UAA'
sys.path.append(module_path)
from UAA_package.NLP_tools import *
from UAA_package.functions import *
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

# Create output directory if it doesn't exist
output_dir = '/Users/jacoponudo/Documents/thesis/src/UAA/output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loading the processed data
social_media_name = "voat"
root = '/Users/jacoponudo/Documents/thesis/'
data = pd.read_csv(root + 'src/PRO/output/' + social_media_name + '_processed.csv')


#3.0 - Estract informations from columns
#* La lunghezza mediana dei testi( numero di parole )
word_count_per_post = data.groupby('user')['text'].apply(lambda x: x.fillna('').apply(lambda text: len(str(text).split()))).reset_index()
lunghezza=word_count_per_post.groupby('user')['text'].median().reset_index()
lunghezza.columns=['user', 'median_length_comment']

#* Numero unico di parole pronunciate dall’utente (post preprocessing) 
word_count_per_user=data.groupby('user')['unique_word_user'].max().reset_index()

#* La media della tossicità dei commenti 
tox_median_per_user=data.groupby('user')['toxicity_score'].max().reset_index()
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




#3.1 - Vocabolary size and Toxicity
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

#3.2 - Concentration of the commenting activity and of the dialogues
data['created_at'] = pd.to_datetime(data['created_at'])
data['period'] = pd.cut(data['created_at'], bins=[pd.Timestamp('2015-01-01'), pd.Timestamp('2020-03-01'), pd.Timestamp('2020-05-01'), pd.Timestamp('2024-01-01')], labels=['pre-COVID', 'COVID', 'post-COVID'])
data['depth']=(data['sequential_number_of_comment_by_user_in_thread']!=1)
results = []

for topic in tqdm(data['topic'].unique()):
    for period in data['period'].unique():
        df = data[(data['topic'] == topic) & (data['period'] == period)]
        
        depth_grouped_data = df.groupby('user')['depth'].sum()
        depth_values = depth_grouped_data.values
        depth_gini_coef = gini_coefficient(depth_values)

        comment_count_grouped_data = df.groupby('user')['comment_id'].count()
        comment_count_values = comment_count_grouped_data.values
        comment_count_gini_coef = gini_coefficient(comment_count_values)

        results.append({'topic': topic, 'period':period, 'depth_gini': depth_gini_coef, 'comment_count_gini': comment_count_gini_coef})

results_df = pd.DataFrame(results)

results_df.set_index('topic', inplace=True)











# Importing necessary libraries
import matplotlib.pyplot as plt
import numpy as np

# Extracting data for plotting
topics = results_df['topic'].unique()
pre_covid_depth_gini = results_df[results_df['period'] == 'pre-COVID']['depth_gini']
covid_depth_gini = results_df[results_df['period'] == 'COVID']['depth_gini']
post_covid_depth_gini = results_df[results_df['period'] == 'post-COVID']['depth_gini']

# Setting the width of the bars
bar_width = 0.25

# Setting the position of the bars on the x-axis
r1 = np.arange(len(topics))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Creating the bar plot for depth_gini
plt.bar(r1, pre_covid_depth_gini, color='skyblue', width=bar_width, edgecolor='grey', label='Pre-COVID')
plt.bar(r2, covid_depth_gini, color='salmon', width=bar_width, edgecolor='grey', label='COVID')
plt.bar(r3, post_covid_depth_gini, color='lightgreen', width=bar_width, edgecolor='grey', label='Post-COVID')

# Adding xticks on the middle of the group bars
plt.xlabel('Topic', fontweight='bold')
plt.xticks([r + bar_width for r in range(len(topics))], topics)

# Adding a legend
plt.legend()

# Showing the plot
plt.title('Gini Coefficients for Depth (Post 2nd)')
plt.ylabel('Gini Coefficient')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


#seconda parte 
# Extracting data for plotting
topics = results_df['topic'].unique()
pre_covid_comment_gini = results_df[results_df['period'] == 'pre-COVID']['comment_count_gini']
covid_comment_gini = results_df[results_df['period'] == 'COVID']['comment_count_gini']
post_covid_comment_gini = results_df[results_df['period'] == 'post-COVID']['comment_count_gini']

# Setting the width of the bars
bar_width = 0.25

# Setting the position of the bars on the x-axis
r1 = np.arange(len(topics))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Creating the bar plot
plt.bar(r1, pre_covid_comment_gini, color='skyblue', width=bar_width, edgecolor='black', label='Pre-COVID')
plt.bar(r2, covid_comment_gini, color='salmon', width=bar_width, edgecolor='black', label='COVID')
plt.bar(r3, post_covid_comment_gini, color='lightgreen', width=bar_width, edgecolor='black', label='Post-COVID')

# Adding xticks on the middle of the group bars
plt.xlabel('Topic', fontweight='bold')
plt.xticks([r + bar_width for r in range(len(topics))], topics)

# Adding a legend
plt.legend()

# Showing the plot
plt.title('Gini Coefficients for Comment Count')
plt.ylabel('Gini Coefficient')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



