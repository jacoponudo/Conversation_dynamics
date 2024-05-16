from tqdm import tqdm
from itertools import product
def metrix(hawkes_process,users):
    for i in (range(len(users))):
        # Observed
        user=users[i]
        ℋ_t=root[root['user']==user]['time']
        ECDF = np.array(F(x_values, ℋ_t))
        
        # Simulated
        SCDF = np.array(F(x_values, hawkes_process.timestamps[i]))
        
        # Evalutation
        KS, p_value = kstest(ECDF, SCDF)
        errors.append(KS)
    return np.mean(errors),np.std(errors)



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


