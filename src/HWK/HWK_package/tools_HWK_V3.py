from tqdm import tqdm
from itertools import product
def metrix(hawkes_process,users,root,ECDF):
    ℋ_t_simulated=[]
    errors=[]
    for i in (range(len(users))):
        ℋ_t_simulated+=list(hawkes_process.timestamps[i])
        # Simulated
    SCDF = np.array(F(x_values,ℋ_t_simulated))
        
    # Evalutation
    KS, p_value = kstest(ECDF, SCDF)
    return KS,p_value



import concurrent.futures
def run_simulation():
    hawkes_process = SimuHawkesExpKernels(adjacency=alphas, decays=betas, baseline=lambdas, end_time=end_time, force_simulation=True, verbose=False)
    hawkes_process.simulate()
    return "Simulazione completata con successo"


'''


theta=[1.0,2.0,0.2]
def f(theta):
    lamda,alpha,beta=theta
    lambdas = [lamda] * n_users
    alphas = [[alpha] * n_users] * n_users
    betas = [[beta] * n_users] * n_users

    end_time = 1.0  # Simulate for 10 seconds
    
    hawkes_process = SimuHawkesExpKernels(adjacency=alphas, decays=betas, baseline=lambdas, end_time=end_time,force_simulation=True)
    hawkes_process.simulate()

    return metrix(hawkes_process)

bounds = [(1, 20), (0, 10), (0.1, 0.6)]
options = {'maxiter': 100, 'disp': True, 'learning_rate': 0.005}
result = optimize.minimize(f, theta, bounds=bounds, options=options)


'''

# Generate a matrix of adjacency from







def generate_weighted_adjacency_matrix(alpha, num_nodes, p):
    """
    Genera una matrice di adiacenza pesata per un grafo casuale con pesi distribuiti normalmente attorno ad alpha.

    Parameters:
    alpha (float): Media della distribuzione normale per i pesi degli archi.
    num_nodes (int): Numero di nodi nel grafo.
    p (float): Probabilità di connessione tra i nodi.

    Returns:
    np.ndarray: Matrice di adiacenza pesata.
    """
    sigma = 1  # Deviazione standard della distribuzione normale per i pesi

    # Step 1: Creazione di un grafo casuale
    G = nx.erdos_renyi_graph(num_nodes, p)  # Grafo casuale con probabilità di connessione p

    # Step 2: Assegnazione dei pesi
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = np.abs(np.random.normal(alpha, sigma))
    for node in G.nodes():
        G.add_edge(node, node, weight=np.abs(np.random.normal(alpha, sigma)))


    # Step 3: Creazione della matrice di adiacenza
    adj_matrix = nx.to_numpy_array(G, weight='weight')

    return adj_matrix



