#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:49:06 2024

@author: jacoponudo
"""

import networkx as nx
active_users = data.groupby('user')['comment_id'].count()
df = data[data['user'].isin(active_users[active_users > 5].index)]
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



degree_sequence = [d for n, d in G.degree()]



# Plotta la distribuzione del grado dei nodi
plt.hist(degree_sequence, bins=10000, alpha=0.7, color='b', edgecolor='black')
plt.title("Distribuzione del grado dei nodi")
plt.xlabel("Grado")
plt.ylabel("Numero di nodi")
plt.xlim(-1,1000)
plt.grid(True)
plt.show()

# fai pruning 
grado_minimo = 30
G_pruned = prune_graph(G, grado_minimo)




blue_subgraph = G_pruned.subgraph(deep_users)
red_subgraph = G_pruned.subgraph(flash_users)

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




# PLOTleaning user leaning amici  (boxplot)


import networkx as nx
dataset_plot = {'x': [], 'y': [],'z':[]}

# Supponiamo che il grafo sia già stato creato e sia memorizzato nella variabile G

number_of_posts=data.groupby('user')['root_submission'].nunique()
for node in tqdm(G.nodes()):
    if not (isinstance(node, float) and math.isnan(node)):
        x = data[data['user'] == node]['user_type'].iloc[0]  # Assumendo che ci sia solo un valore per x per nodo
        neighbors = list(G.neighbors(node))
        y = (data[data['user'].isin(neighbors)]['user_type'] == 'deep').mean()
        z = number_of_posts.get(node, 0)
        
        dataset_plot['x'].append(x)
        dataset_plot['y'].append(y)
        dataset_plot['z'].append(z)

# Convertire il dizionario in un DataFrame pandas
df_plot = pd.DataFrame(dataset_plot)

# Creare il boxplot
plt.figure(figsize=(8, 6))
df_plot.boxplot(column='y', by='x',showfliers=False)
plt.title('Boxplot di y per i diversi valori di x')
plt.xlabel('x')
plt.ylabel('y')
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