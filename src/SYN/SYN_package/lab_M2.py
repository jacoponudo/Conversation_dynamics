# Parameters 

dict_parameter={'gb': {'gamma': 3.3581436231409034,
  'a': 0.5823839228754921,
  'b': 30.15833332622568,
  'loc': 0.0009999999999999998,
  'scale': 5.624393766327099,
  'alpha': 0.3,
  'lambda': 0.20000000000000004,
  'c': 1.153601162254783,
  'd': 0.12145921455196622,
  'l': 2.7777777777777775e-06,
  's': 1.2437363896183875,
  'cf': 0.8102059793381664,
  'df': 0.1960299987299946,
  'lf': 2.7777777777777775e-06,
  'sf': 1.539098221807941,
  'ka': 1.311812182091273,
  'kb': 1.3860013045827573,
  'kloc': 0.0011829628940588704,
  'kscale': 0.9959070877390364},
 'rd': {'gamma': 2.1172386372477376,
  'a': 1.3208418726992361,
  'b': 366274394.2558266,
  'loc': 0.00043188573340450587,
  'scale': 21377429.04281839,
  'alpha': 0.3,
  'lambda': 0.5000000000000001,
  'c': 1.4035122312087873,
  'd': 0.34547543092801714,
  'l': 2.777777777776469e-06,
  's': 0.009685929534020282,
  'cf': 1.061267002974987,
  'df': 0.16381176822206486,
  'lf': 2.7777777777777775e-06,
  'sf': 1.708837260306469,
  'ka': 0.98131435112003,
  'kb': 0.8452499045619575,
  'kloc': 0.0022905406038342627,
  'kscale': 0.9934541402472297},
 'fb': {'gamma': 1.920612852062878,
  'a': 0.3844295492882861,
  'b': 294.47288219865607,
  'loc': 0.0009999999999999998,
  'scale': 33.07578198818712,
  'alpha': 0.1,
  'lambda': 1.0500000000000003,
  'c': 1.4925263821442911,
  'd': 0.08849381153322906,
  'l': 2.7777777777777775e-06,
  's': 1.3440750553651393,
  'cf': 1.1530515610823424,
  'df': 0.1221286925154284,
  'lf': 2.7777777777777775e-06,
  'sf': 1.243471290490271,
  'ka': 1.5472072908627337,
  'kb': 1.4924241515306047,
  'kloc': -2.313415765298187e-05,
  'kscale': 0.9989946703644328},
 'vo': {'gamma': 4.765796666436497,
  'a': 0.7132950528871953,
  'b': 15.52622890706321,
  'loc': 0.0009999999999999998,
  'scale': 1.1164476880697753,
  'alpha': 0.3,
  'lambda': 0.5000000000000001,
  'c': 1.5771371987815774,
  'd': 0.25332997728949264,
  'l': 2.7777777777777775e-06,
  's': 0.0185829545134503,
  'cf': 1.1877604878563408,
  'df': 0.1432213467957256,
  'lf': 2.7777777777777775e-06,
  'sf': 1.5169432923443469,
  'ka': 1.5398246083070783,
  'kb': 1.33432679259424,
  'kloc': 0.0030737584032413234,
  'kscale': 0.9933356299071279}}

parameters=dict_parameter['rd']


gamma=parameters['gamma']
a=parameters['a']
b=parameters['b']
loc=parameters['loc']
scale=parameters['scale']
alpha=parameters['alpha']
lambda_=parameters['lambda']
c=parameters['c']
d=parameters['d']
l=parameters['l']
s=parameters['s']
cf=parameters['cf']
df=parameters['df']
lf=parameters['lf']
sf=parameters['sf']
ka=parameters['ka']
kb=parameters['kb']
kloc=parameters['kloc']
kscale=parameters['kscale']
gamma=parameters['gamma']
a=parameters['a']
b=parameters['b']
loc=parameters['loc']
scale=parameters['scale']
alpha=parameters['alpha']
min_users=50

# Functions
import random
def last_value_not_na(lista):
    for valore in reversed(lista):
        if not math.isnan(valore):
            return valore
    return None

def first_nan(lista):
    for i, valore in enumerate(lista):
        if math.isnan(valore):
            return i
    return -1  # Se non c'è nessun NaN nella lista


# Code
number_of_users = int(np.round(simulate_number_of_users(gamma, min_users, size=1)[0]))
T0s = simulate_initial_comment(a, b, loc, scale, size=number_of_users)
Ns = simulate_number_of_comments(alpha, lambda_, number_of_users) + 1
thread = [[T0s[i]] + [np.nan] * (Ns[i] - 1) for i in range(number_of_users)]


while any(np.isnan(value) for sublist in thread for value in sublist):
    for i,interaction in enumerate(thread):
        j=first_nan(interaction)
        if j!=-1:
            view,lag=burr.rvs(c, d, l, s, size=2)[0:2]
            lag=lag/100
            alpha=random.random()
            if alpha>0.9:
                last_comments = [last_value_not_na(lista) for lista in thread]
                filtered_values = [value for value in last_comments if interaction[j-1] < value <= interaction[j-1] + view]
            else:
                exclude_first_comment = [sublist for sublist in thread if len(sublist) != 1]
                last_comments = [last_value_not_na(lista) for lista in exclude_first_comment]
                filtered_values = [value for value in last_comments if interaction[j-1] < value ]
            if len(filtered_values)!=0:
                sampled_value = random.choice(filtered_values)
                if j<(len(interaction)-1):
                    thread[i][j]=float(sampled_value+lag )
                else:
                    thread[i][j]=float(sampled_value+ lag)
    
            else: 
                if j<(len(interaction)-1):
                    thread[i][j]=float(interaction[j-1]+ lag)
                else:
                    thread[i][j]=float(interaction[j-1]+ lag)
        
thread = [[min(1, value) for value in sublist] for sublist in thread]





valori_uguali_a_1 = [valore >1  for sublist in thread for valore in sublist ]

# Calcola la media dei valori uguali a 1
if valori_uguali_a_1:
    media = sum(valori_uguali_a_1) / len(valori_uguali_a_1)

    
print(media)



# Creazione di una lista di dizionari con valore e ID della lista
data = []
for i, sublist in enumerate(thread):
    for j, value in enumerate(sublist):
        data.append({
            'value': value, 
            'list_id': i,
            'first_comment': j == 0
        })

# Creazione del DataFrame
df = pd.DataFrame(data)


# Ordinamento del DataFrame per 'value'
df_sorted = df.sort_values(by='value').reset_index(drop=True)

# Divisione del DataFrame in 10 subdataset
subdatasets = np.array_split(df_sorted, 20)

# Conta dei valori unici di 'list_id' in ogni subdataset
list_id_counts = [np.mean(subset['first_comment']) for subset in subdatasets]

print(list_id_counts)


def is_decreasing_trend(counts):
    diffs = np.diff(counts)
    return np.mean(diffs <= 0)

# Verifica del trend decrescente
decreasing_trend = is_decreasing_trend(list_id_counts)

print("Decreasing trend:", decreasing_trend)
