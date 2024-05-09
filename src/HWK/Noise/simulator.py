
len(ℋ_t) # il valore reale di  commenti
𝛉_exp # il valore stimato dei parametri
θ_exp_mle_T # il valore stimato dei parametri considerando tutte le conversazioni degli utenti

simulazioni_su_thread=[]
simulazioni_su_utente=[]
for i in range(10000):
    simulazione_su_utente=exp_simulate_by_composition_alt(𝛉_exp, max(ℋ_t)+mean_lag)
    simulazioni_su_utente.append(len(simulazione_su_utente))
    simulazione_su_thread=exp_simulate_by_composition_alt(𝛉_exp, max(ℋ_t)+mean_lag)
    simulazioni_su_thread.append(len(simulazione_su_thread))
    

print(np.mean(simulazioni_su_utente)) # il numero di commenti  atteso simulando con la prima stima  dei parametri
print(np.mean(simulazioni_su_thread)) # il numero di commenti  atteso simulando con la seconda stima  dei parametri
