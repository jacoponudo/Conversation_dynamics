
# Jacopo's functions for fitting hawkes process

import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from scipy.stats import chi2

#### Max Likelihood Estiamtion #### 
# The following are 3 functions to estimate, given the realizations of a process, the parameters:
# -individual_exp_log_likelihood: calculates the individual log likelihood givene an observation.
# -sum_individual_exp_log_likelihood: sum all the likelihood functions on different observations.
# -considering the previous function as the objective function minimize the loss, with respect to the combination of parameters. 
    

def individual_exp_log_likelihood(event_times, T, parameters):
    """
    Calculate the individual log-likelihood of exponential survival model for a single event sequence.

    Args:
        event_times (array-like): Time of events.
        T (float): Total observation time.
        parameters (tuple): Tuple of parameters (λ, α, β).

    Returns:
        float: Log-likelihood of the given event sequence.
    """
    λ, α, β = parameters
    times = event_times
    N_T = len(times)

    A = np.empty(N_T, dtype=np.float64)
    A[0] = 0
    for i in range(1, N_T):
        A[i] = np.exp(-β * (times[i] - times[i - 1])) * (1 + A[i - 1])

    likelihood = -λ * T
    for i, t_i in enumerate(event_times):
        likelihood += np.log(λ + (α) * A[i]) - ((α) / β) * (1 - np.exp(-β * (T - t_i)))
    return likelihood


def sum_individual_exp_log_likelihood(event_sequences, T, parameters):
    """
    Calculate the sum of individual log-likelihoods of exponential survival model for multiple event sequences.

    Args:
        event_sequences (list): List of event sequences.
        T (float): Total observation time.
        parameters (tuple): Tuple of parameters (λ, α, β).

    Returns:
        float: Sum of log-likelihoods of the given event sequences.
    """
    total_likelihood = 0
    num_sequences = len(event_sequences)
    for i in (range(num_sequences)):
        likelihood = individual_exp_log_likelihood(event_sequences[i], T, parameters)
        total_likelihood += likelihood
    return total_likelihood


def exponential_mle(event_times, T, initial_parameters=np.array([1.0, 2.0, 3.0])):
    """
    Estimate Maximum Likelihood Estimates (MLE) for parameters of the exponential survival model.

    Args:
        event_times (array-like): Time of events.
        T (float): Total observation time.
        initial_parameters (array-like, optional): Initial guess for parameters. Defaults to [1.0, 2.0, 3.0].

    Returns:
        array-like: MLE estimates for parameters (λ, α, β).
    """
    eps = 1e-5
    parameter_bounds = ((eps, None), (eps, None), (eps, None))
    loss_function = lambda parameters: -sum_individual_exp_log_likelihood(event_times, T, parameters)
    mle_parameters = minimize(loss_function, initial_parameters, bounds=parameter_bounds).x
    return np.array(mle_parameters)

#### Max Likelihood Estiamtion + Toxicity parameter #### 
# The following are 3 functions to estimate, given the realizations of a process, the parameters:
# -individual_exp_log_likelihood: calculates the individual log likelihood givene an observation.
# -sum_individual_exp_log_likelihood: sum all the likelihood functions on different observations.
# -considering the previous function as the objective function minimize the loss, with respect to the combination of parameters (considering also the alpha parameter of excitement driven by toxicity). 

def individual_exp_log_likelihood_toxicity(event_times, magnitude, T, parameters):
    """
    Calculate the individual log-likelihood of exponential survival model considering toxicity for a single event sequence.

    Args:
        event_times (array-like): Time of events.
        magnitude (array-like): Magnitude of toxicity for each event.
        T (float): Total observation time.
        parameters (tuple): Tuple of parameters (λ, α, α_T, β).

    Returns:
        float: Log-likelihood of the given event sequence considering toxicity.
    """
    λ, α, α_T, β = parameters
    times = event_times
    N_T = len(times)

    A = np.empty(N_T, dtype=np.float64)
    A[0] = 0
    for i in range(1, N_T):
        A[i] = np.exp(-β * (times[i] - times[i - 1])) * (1 + A[i - 1])

    likelihood = -λ * T
    for i, t_i in enumerate(event_times):
        mi = magnitude[i]
        likelihood += np.log(λ + (α + α_T * mi) * A[i]) - ((α + α_T * mi) / β) * (1 - np.exp(-β * (T - t_i)))
    return likelihood


def sum_individual_exp_log_likelihood_toxicity(event_sequences, magnitude_sequences, T, parameters):
    """
    Calculate the sum of individual log-likelihoods of exponential survival model considering toxicity for multiple event sequences.

    Args:
        event_sequences (list): List of event sequences.
        magnitude_sequences (list): List of magnitude sequences corresponding to event sequences.
        T (float): Total observation time.
        parameters (tuple): Tuple of parameters (λ, α, α_T, β).

    Returns:
        float: Sum of log-likelihoods of the given event sequences considering toxicity.
    """
    total_likelihood = 0
    num_sequences = len(event_sequences)
    for i in (range(num_sequences)):
        likelihood = individual_exp_log_likelihood_toxicity(event_sequences[i], magnitude_sequences[i], T, parameters)
        total_likelihood += likelihood
    return total_likelihood


def exponential_mle_toxicity(event_times, magnitude, T, initial_parameters=np.array([0.5, 0, 0, 0])):
    """
    Estimate Maximum Likelihood Estimates (MLE) for parameters of the exponential survival model considering toxicity.

    Args:
        event_times (array-like): Time of events.
        magnitude (array-like): Magnitude of toxicity for each event.
        T (float): Total observation time.
        initial_parameters (array-like, optional): Initial guess for parameters. Defaults to [1.0, 2.0, 5.0, 3.0].

    Returns:
        array-like: MLE estimates for parameters (λ, α, α_T, β).
    """
    eps = 1e-30
    parameter_bounds = ((eps, None), (eps, None), (eps, None), (eps, None))
    loss_function = lambda parameters: -sum_individual_exp_log_likelihood_toxicity(event_times, magnitude, T, parameters)
    mle_parameters = minimize(loss_function, initial_parameters, bounds=parameter_bounds).x
    return np.array(mle_parameters)


#### Data Preparation ####
# Function to prepare data, given: data from user, data from all users, and list of conversations to focus on.
# return ℋ_T_list (list of list with timestamp of each single comment in a process standarsized among 0 and 1), magnitude_list (list of list of manginute of toxicity for each comment.) 

def prepare_data(df, dataset):
    """
    Filter toxic comments based on selected conversations and prepare data.

    Args:
    - df (DataFrame): DataFrame containing the user data
    - dataset (DataFrame): Complete dataset containing all conversations

    Returns:
    - ℋ_T_list (list): List of standardized conversation timestamps
    - magnitude_list (list): List of toxicity scores
    """

    # Filter toxic comments based on selected conversations
    toxic=df
    ℋ_T_list = []
    magnitude_list = []
    for conversation in (toxic['root_submission'].unique()):
        all_chat = np.array([np.datetime64(x.replace(tzinfo=None)).astype(np.int64) for x in dataset[dataset['root_submission'] == conversation]['created_at']])
        single_conversation = toxic[toxic['root_submission'] == conversation].copy()
        observed_data = np.array([np.datetime64(x.replace(tzinfo=None)).astype(np.int64) for x in single_conversation['created_at']])
        min_value, max_value = all_chat.min(), all_chat.max()
        standardized_data = (observed_data - min_value) / (max_value - min_value)
        standardized_data.sort()
        magnitude = np.array(single_conversation['toxicity_score'])
        ℋ_T_list.append(standardized_data)
        magnitude_list.append(magnitude)

    return ℋ_T_list, magnitude_list

def filter_dataset(dataset, username, min_comments,sample):
    """
    Filter dataset for a specific user with a minimum number of comments.

    Args:
        dataset (DataFrame): Input dataset.
        username (str): Username to filter.
        min_comments (int): Minimum number of comments.

    Returns:
        DataFrame: Filtered dataset.
    """
    # Filter data for the specified user
    df = dataset[dataset['user'] == username]
    
    # Group conversations by root_submission and count the number of comments
    grouped_data = df.groupby('root_submission').size().reset_index(name='num_comments')
    
    # Filter conversations with more than min_comments
    if sample==False:
        conversations = grouped_data[grouped_data['num_comments'] > min_comments]
    else:
        conversations = grouped_data[grouped_data['num_comments'] > min_comments].sample(sample,replace=True)
    
    # Filter toxic comments based on selected conversations
    data = df[df['root_submission'].isin(conversations['root_submission'])]
    
    return data


#### Model fitting evalution ####
def calculate_aic(n, k, likelihood):
    """
    Calculate the Akaike Information Criterion (AIC) for a given likelihood and number of parameters.

    Args:
        n (int): Number of observations.
        k (int): Number of parameters.
        likelihood (float): Log-likelihood of the model.

    Returns:
        float: AIC value.
    """
    return 2 * k - 2 * likelihood

def calculate_bic(n, k, likelihood):
    """
    Calculate the Bayesian Information Criterion (BIC) for a given likelihood, number of parameters, and number of observations.

    Args:
        n (int): Number of observations.
        k (int): Number of parameters.
        likelihood (float): Log-likelihood of the model.

    Returns:
        float: BIC value.
    """
    return k * np.log(n) - 2 * likelihood



#### Select subsample of users ####

def select_users_with_multiple_comments(data, min_comments_per_post=3, min_post_count=3):
    """
    Selects users who have made more than 'min_comments_per_post' comments under the same post
    for at least 'min_post_count' times.

    Args:
    - data: DataFrame containing the data
    - min_comments_per_post: Minimum number of comments under the same post to be considered
    - min_post_count: Minimum number of posts that satisfy the above criteria

    Returns:
    - DataFrame containing the users who satisfy the criteria
    """
    # Calculate the count of comments for each user and post
    comment_count = data.groupby(['user', 'root_submission']).size().reset_index(name='comment_count')

    # Filter the results for users who have made more than 'min_comments_per_post' comments under the same post
    filtered_data = comment_count[comment_count['comment_count'] > min_comments_per_post]

    # Select only the users who have satisfied the criteria for at least 'min_post_count' times
    final_result = filtered_data.groupby('user').filter(lambda x: len(x) >= min_post_count)['user'].unique()


    return final_result

#### Perform analysis, fitting the model ####
# Given a sample of users, we fit the simple Hawkes process and the version with the alpha of toxicity. 

def analyze_users(dataset, sample_users, min_comments):
    """
    Analyzes users based on a given dataset, a sample of users, and a minimum number of comments.

    Parameters:
        dataset (DataFrame): The dataset containing user comments.
        sample_users (list): A list of users to analyze.
        min_comments (int): The minimum number of comments required for analysis.

    Returns:
        list: A list of dictionaries containing the analysis results for each user.
    """
    user_analysis = []
    
    for user in tqdm(sample_users):
        # Filter comments by the user with at least the minimum number of comments
        toxic = filter_dataset(dataset, user, min_comments=min_comments, sample=False)
        
        # Prepare data for each conversation including timestamp of events and magnitude for each comment
        ℋ_T_list, magnitude_list = prepare_data(toxic, dataset)
        
        # Estimate parameters for toxicity model
        θ_exp_mle_T = exponential_mle_toxicity(ℋ_T_list, magnitude_list, 1)
        
        # Estimate parameters for simple model
        θ_exp_mle = exponential_mle(ℋ_T_list, 1)
        
        # Compute likelihood values for both models
        Lh_simple_model = sum_individual_exp_log_likelihood(ℋ_T_list, 1, θ_exp_mle)
        Lh_toxicity_model = sum_individual_exp_log_likelihood_toxicity(ℋ_T_list, magnitude_list, 1, θ_exp_mle_T)
        
        # Calculate size parameter alpha_T0 for the toxicity model
        size_alpha_T0 = θ_exp_mle_T[2] - θ_exp_mle_T[1] / θ_exp_mle_T[1]
        
        # Calculate AIC values for both models
        n = len(ℋ_T_list)
        k_T = len(θ_exp_mle_T)
        k = len(θ_exp_mle)
        aic_T = calculate_aic(n, k, Lh_toxicity_model)
        aic = calculate_aic(n, k_T, Lh_simple_model)
        
        # Calculate likelihood ratio and p-value
        log_likelihood_full = -sum_individual_exp_log_likelihood_toxicity(ℋ_T_list, magnitude_list, 1, θ_exp_mle_T)
        log_likelihood_simple = -sum_individual_exp_log_likelihood(ℋ_T_list, 1, θ_exp_mle)
        LR = 2 * (log_likelihood_full - log_likelihood_simple)
        df = len(θ_exp_mle_T) - len(θ_exp_mle)
        p_value = 1 - chi2.cdf(LR, df)
        
        # Check significance of likelihood test
        if p_value < 0.05:
            significance = "The full model provides better data fit than the simpler model (p-value < 0.05)"
        else:
            significance = "The full model does not provide better data fit than the simpler model (p-value >= 0.05)"
        
        # Append user analysis data to list
        user_analysis.append({'User': user,
                              'Lambda':θ_exp_mle[0],
                              'Lambda_T':θ_exp_mle_T[0],
                              'Alpha_1':θ_exp_mle[1],
                              'Alpha_1_T':θ_exp_mle_T[1],
                              'Alpha_2_T':θ_exp_mle_T[2],
                              'Beta':θ_exp_mle[2],
                              'Beta_T':θ_exp_mle_T[3],
                              'AIC_Toxicity_Model': aic_T,
                              'AIC_Simple_Model': aic,
                              'Number_of_Comments': len(toxic),
                              'P_Value_Likelihood_Test': p_value,
                              'Significance': significance})
    
    return user_analysis


