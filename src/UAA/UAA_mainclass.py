# Notes for Jacopo:
''' 

'''

# Define the module path + import packages
import sys
module_path = '/Users/jacoponudo/Documents/thesis/src/UAA'
sys.path.append(module_path)
from UAA_package import *


#- [ ] Quanti commenti appartengono a un dialogo (almeno due commenti per utente) e quanti a un apparizione unica. 

# Set source
source_data='/Users/jacoponudo/Downloads/voat_labeled_data_unified.parquet'
root='/Users/jacoponudo/Documents/thesis/'
output=root+'src/UAA/output'

# Read the dataset
dataset = pd.read_parquet(source_data)

# For each comment analyze if belong to a dialogue (more than 1 comment from user in the thread) or not. 
# Group by 'user' and 'root_submission' and count the number of comments
dataset['comment_count'] = dataset.groupby(['user', 'root_submission'])['user'].transform('count')
# Create a new column 'multiple_comments' with True or False values based on the condition
dataset['multiple_comments'] = dataset['comment_count'] > 1

# Display the updated dataset with the new column
print(merged_dataset)