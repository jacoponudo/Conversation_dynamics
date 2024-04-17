import numpy as np
import tick
from tick.plot import plot_point_process

# Generazione dei dati casuali (ad esempio, tempi di arrivo delle conversazioni)
# Questo è solo un esempio di dati casuali, sostituiscilo con i tuoi dati reali
n_points = 1000
timestamps = np.sort(np.random.uniform(0, 100, n_points))

# Creazione di un oggetto PointProcess dalla serie temporale
point_process = tick.hawkes.SimuHawkesExpKernels(adjacency=[[0.1]], decays=[2.0])
point_process.simulate([[timestamps]])

# Plot dei dati puntuali
fig, ax = plot_point_process(point_process.timestamps[0], plot_options={'c': 'b', 'label': 'Dati'})

# Fit del processo di Hawkes
hawkes_model = tick.hawkes.HawkesExpKern(decay=2.0)
hawkes_model.fit(point_process.timestamps)

# Plot del modello di Hawkes
fig, ax = hawkes_model.plot_intensity(ci=0.95, legend=True, plot_options={'c': 'r', 'label': 'Modello di Hawkes'})

# Mostra il grafico
ax.legend(loc='best')
plt.xlabel('Tempo')
plt.ylabel('Intensità')
plt.title('Processo di Hawkes per le conversazioni')
plt.show()
