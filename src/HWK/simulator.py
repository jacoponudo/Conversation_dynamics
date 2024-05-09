
len(ℋ_t)
simulazione=exp_simulate_by_composition_alt(𝛉_exp, max(ℋ_t)+mean_lag)
len(simulazione)


simulazioni=[]
for i in range(1000000):
    simulazione=exp_simulate_by_composition_alt(𝛉_exp, max(ℋ_t)+mean_lag)
    simulazioni.append(len(simulazione))
    
np.mean(simulazioni)
