# Brurst analysis: controlla se nelle conversazioni quando si verifica una tempesta di commenti si è in presenza di un maggiore livello di tossicità.

import matplotlib.pyplot as plt
import numpy as np
def plot_bursts(t, bursts):
    """
    Visualizza la serie temporale degli eventi con i burst evidenziati
    t: array di tempi degli eventi
    bursts: array di tempi dei burst
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, [1] * len(t), 'o', color='blue', markersize=5, label='Eventi')
    plt.plot(bursts, [1] * len(bursts), 'o', color='red', markersize=3, label='Bursts')
    plt.xlabel('Tempo')
    plt.ylabel('Evento')
    plt.title('Serie temporale degli eventi con burst evidenziati')
    plt.legend()
    plt.show()

def kleinberg_burst_detection(t, alpha, beta, gamma, window_size, burst_percentage):
    """
    Kleinberg Burst Detection Algorithm with adaptive burst percentage and wider windows
    t: array of event times
    alpha: normalization parameter
    beta: temporal decay parameter
    gamma: aging parameter
    window_size: size of the window for burst probability calculation
    burst_percentage: desired percentage of events to classify as bursts (e.g., 0.1 for 10%)
    """
    n = len(t)
    bursts = []
    p = [0] * n  # Initialize burst probability

    # Calculate burst probability for each time t_i
    for i in range(n):
        for j in range(max(0, i - window_size), i):  # Use wider window for probability calculation
            dt = t[i] - t[j]
            p[i] += alpha * beta**dt * (1 - gamma)**(t[i] - t[j] - 1)

    # Determine threshold for burst detection
    threshold = sorted(p, reverse=True)[int(burst_percentage * n)]

    # Detect bursts based on the threshold
    for i in range(1, n-1):
        if p[i] > threshold and p[i] > p[i-1] and p[i] > p[i+1]:
            bursts.append(t[i])

    return bursts

# Funzione per dividere i dati in burst e non-burst
def divide_into_bursts_and_non_bursts(t, bursts):
    """
    Divide i dati standardizzati in burst e non-burst
    t: array di tempi degli eventi
    bursts: array di tempi dei burst
    Restituisce due array, uno per i burst e uno per i non-burst
    """
    burst_indices = np.searchsorted(t, bursts)  # Trova gli indici dei burst nei dati
    all_indices = np.arange(len(t))
    non_burst_indices = np.setdiff1d(all_indices, burst_indices)  # Trova gli indici dei non-burst
    
    # Dividi i dati in burst e non-burst
    bursts_data = t[burst_indices]
    non_bursts_data = t[non_burst_indices]
    
    return bursts_data, non_bursts_data


def add_burst_column(df, standardized_data, bursts):
    """
    Aggiunge una colonna al DataFrame indicando se ciascuna riga corrisponde a un burst o no
    df: DataFrame originale
    standardized_data: array di tempi degli eventi standardizzati
    bursts: array di tempi dei burst
    Restituisce il DataFrame con la nuova colonna aggiunta
    """
    burst_indices = np.searchsorted(standardized_data, bursts)  # Trova gli indici dei burst nei dati
    df['burst'] = False  # Inizializza tutti i valori a False
    
    # Imposta a True i valori corrispondenti ai burst
    df.loc[df.index[burst_indices], 'burst'] = True
    
    return df
