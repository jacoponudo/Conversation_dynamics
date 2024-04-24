# Notes for Jacopo:
''' 

'''




# Define the module path + import packages
import sys
module_path = '/Users/jacoponudo/Documents/thesis/src/HWK'
sys.path.append(module_path)
import numpy as np
import pandas as pd
from tqdm import tqdm
from HWK_package.functions import *
from scipy import stats
import random 
from scipy.stats import chi2


# Read the dataset
dataset = pd.read_parquet('/Users/jacoponudo/Downloads/voat_labeled_data_unified.parquet')

sample_users=select_users_with_multiple_comments(dataset, min_comments_per_post=3, min_post_count=3).sample(100)




user_analysis=analyze_users(dataset, sample_users, 3)

user_analysis=pd.DataFrame(user_analysis)




#grafico
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
