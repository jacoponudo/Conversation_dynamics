def create_hawkes_parameters(N,l,b):
    # Tassi base per i N processi
    lambda0 = np.ones(N)*l

    # Intensità dell'effetto dei processi
    alpha = np.ones((N, N)) - np.eye(N)

    # Parametri di decadimento per i N processi
    beta = np.ones(N) * b

    return lambda0, alpha, beta

def simulate_hawkes_dependent(T, lambda0, alpha, beta,events):
    # Vettori per memorizzare i tempi degli eventi
    events =  events# Eventi iniziali
    time = max([max(event_list) for event_list in events])
    
    # Simulazione degli eventi fino al tempo T
    while time < T:
        # Calcolo dell'intensità attuale per ciascun processo
        lambda_t = np.zeros(N)
        for i in range(N):
            lambda_t[i] = lambda0[i]
            for j in range(N):
                lambda_t[i] += sum(alpha[i, j] * np.exp(-beta[j] * (time - np.array(events[j]))))
        
        
        for i in range(N):
            if dizionario[i]==len(events[i]):
                lambda_t[i]=0
        # Calcolo del tasso totale
        lambda_total = np.sum(lambda_t)
        
        if lambda_total==0:
            break
        # Tempo fino al prossimo evento
        time_to_next_event = np.random.exponential(1 / lambda_total)
        
        # Aggiornamento del tempo corrente
        time += time_to_next_event
        
        if time >= T:
            break
        
        # Determinazione del tipo di evento
        event_type = np.random.choice(list(range(N)), p=lambda_t / lambda_total)
        events[event_type].append(time)
    
    # Ritorno dei tempi degli eventi
    return events



def divide_in_gruppi(lista, dimensione):
    # Inizializza una lista vuota per contenere i gruppi
    gruppi = []
    
    # Calcola il numero di gruppi necessari
    num_gruppi = len(lista) // dimensione
    
    # Itera attraverso la lista e aggiungi gruppi di dimensione specificata alla lista di gruppi
    for i in range(num_gruppi):
        gruppi.append(lista[i * dimensione:(i + 1) * dimensione])
    
    # Se ci sono elementi rimanenti, aggiungi l'ultimo gruppo
    if len(lista) % dimensione != 0:
        gruppi.append(lista[num_gruppi * dimensione:])
    
    return gruppi


def simulate_data_H(social, parameters, num_threads=False, activate_tqdm=True, min_users=50):
    gamma=parameters['gamma']
    a=parameters['a']
    b=parameters['b']
    loc=parameters['loc']
    scale=parameters['scale']
    alpha=parameters['alpha']
    lambda_=parameters['lambda']
    data = []
    if num_threads:
        num_threads = min(num_threads, len(social['post_id'].unique()))
        thread_ids = random.sample(list(social['post_id'].unique()), num_threads)
    else:
        thread_ids = social['post_id'].unique()
    
    if activate_tqdm:
        thread_ids = tqdm(thread_ids)
    
    for th in thread_ids:
        thread = social[social['post_id'] == th]
        number_of_users = int(np.round(simulate_number_of_users(gamma, min_users, size=1)[0]))
        T0s = simulate_initial_comment(a, b, loc, scale, size=number_of_users)
        Ns=simulate_number_of_comments(alpha, lambda_,number_of_users)
        thread = [[T0s[i]] + [0] * (Ns[i] - 1) for i in range(number_of_users)]
        size_clique = 5  # Dimensione desiderata di ogni gruppo
        cliques = divide_in_gruppi(thread, size_clique)
        final=[]
        for clique  in cliques:
            # Replace nan with 0
            lista = [[0 if math.isnan(x) else x for x in sublist] for sublist in clique]
            dizionario = {i: len(sotto_lista) for i, sotto_lista in enumerate(lista)}
            N = len(lista)
            lambda0, alpha, beta = create_hawkes_parameters(N,12,0.04)
            
            events = [[sublist[0]] for sublist in lista]
            
            
            # Simulazione per un intervallo di tempo T
            T = 1
            events = simulate_hawkes_dependent(T, lambda0, alpha, beta,events)
            for interaction in events:
                for j, t in enumerate(interaction):
                    data.append({'user_id': f'User_{i}', 'post_id': th, 'temporal_distance_birth_base_100h': t, 'sequential_number_of_comment_by_user_in_thread': j + 1})

    simulated = pd.DataFrame(data)
    observed = social[social['post_id'].isin(simulated['post_id'].unique())][['user_id', 'post_id', 'temporal_distance_birth_base_100h', 'sequential_number_of_comment_by_user_in_thread']]

    return simulated, observed