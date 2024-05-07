import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests


data = pd.read_parquet('/Users/jacoponudo/Documents/thesis/data/voat/voat_labeled_data_unified.parquet')

# Seleziona la conversazione specifica
conversation_id = '3357323'
conversation_data = data[data['root_submission'] == conversation_id]

# Conta il numero di commenti fatti da ciascun utente nella conversazione
sample = data.groupby(['user','root_submission']).size()

# Seleziona solo gli utenti che hanno fatto più di 5 commenti nella conversazione
users_with_more_than_5_comments = sample[sample > 5].index







data[data['root_submission']=='3054012'].user


np.random.seed(0)
activity_user_a = list(data[(data['user']=='zyklon_b') & (data['root_submission']=='3054012')]['created_at'])
activity_user_b = list(data[(data['user']=='NiggerVirus') & (data['root_submission']=='3054012')]['created_at'])

activity_user_a.sort()
activity_user_b.sort()

serie_temporale_1 = activity_user_a
serie_temporale_2 = activity_user_b

from datetime import timedelta

# Definisci una soglia temporale per determinare se due timestamp sono considerati "connessi"
soglia_temporale = timedelta(minutes=5)

# Itera su ogni timestamp nella serie temporale 2 e verifica se esiste un corrispondente nella serie temporale 1 entro la soglia temporale
for timestamp_2 in serie_temporale_2:
    for timestamp_1 in serie_temporale_1:
        if abs(timestamp_2 - timestamp_1) <= soglia_temporale:
            print("Connessione trovata tra", timestamp_2, "e", timestamp_1)
            break 



activity_user_a = list(data[(data['user']=='zyklon_b') & (data['root_submission']=='3054012')]['toxicity_score'])
activity_user_b = list(data[(data['user']=='NiggerVirus') & (data['root_submission']=='3054012')]['toxicity_score'])
activity_user_a = list(data[(data['user']=='zyklon_b') & (data['root_submission']=='3054012')]['created_at'])
activity_user_b = list(data[(data['user']=='NiggerVirus') & (data['root_submission']=='3054012')]['created_at'])

import matplotlib.pyplot as plt
import pandas as pd

# Trasformo le variabili in DataFrame
data_a = pd.DataFrame({'time':  list(data[(data['user']=='zyklon_b') & (data['root_submission']=='3054012')]['created_at']), 'toxicity_score': list(data[(data['user']=='zyklon_b') & (data['root_submission']=='3054012')]['toxicity_score'])})
data_b = pd.DataFrame({'time':  list(data[(data['user']=='NiggerVirus') & (data['root_submission']=='3054012')]['created_at']), 'toxicity_score': list(data[(data['user']=='NiggerVirus') & (data['root_submission']=='3054012')]['toxicity_score'])})

# Plot dello scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(data_a['time'], data_a['toxicity_score'], color='blue', label='User A', alpha=0.5)
plt.scatter(data_b['time'], data_b['toxicity_score'], color='red', label='User B', alpha=0.5)

# Aggiunta di etichette e titoli
plt.title('Scatterplot tra Attività Utente A e Utente B')
plt.xlabel('Tempo')
plt.ylabel('Punteggio di Tossicità')
plt.legend()

# Mostra il plot
plt.show()


