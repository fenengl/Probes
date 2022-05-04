import langmuir as l
import numpy as np
import scipy.constants as sc
import pandas as pd
from itertools import count
from tqdm import tqdm
"""
PART 1: GENERATE SYNTHETIC DATA USING LANGMUIR
"""

def rand_uniform(N, range):
    """Generate N uniformly distributed numbers in range"""
    return range[0]+(range[1]-range[0])*np.random.rand(N)

def rand_log(N, range):
    """Generate N logarithmically distributed numbers in range"""
    x = rand_uniform(N, np.log(range))
    return np.exp(x)

def random_synthetic_data(N,geo1,geo2,model1, model2,Vs_geo1,Vs_geo2,geometry):

    # def calc_eta(V,T):
    #     eta=(sc.elementary_charge*V)/(sc.Boltzmann*T)
    #     return eta

    ns  =rand_log(N, [4e10, 3e11]) # densities
    Ts = rand_uniform(N, [300, 2800]) # temperatures### or 800
    V0s =rand_uniform(N, [-2,  0]) # floating potentials


    # Generate probe currents corresponding to plasma parameters
    Is_geo1 = np.zeros((N,len(Vs_geo1)))
    Is_geo2 = np.zeros((N,len(Vs_geo2)))


    for i, n, T, V0 in zip(count(), ns, Ts, tqdm(V0s)):
        Is_geo1[i] = model1(geo1, l.Electron(n=n, T=T),V=V0+Vs_geo1)
        Is_geo2[i] = model2(geo2, l.Electron(n=n, T=T), V=V0+Vs_geo2)
        Is=np.append(Is_geo1,Is_geo2,axis=1)
    print(Is)
    Is_cols = ["Is_{0}".format(x) for x in range(Is.shape[1])]
    data=np.append(np.array([ns,Ts,V0s]).T,Is,axis=1)
    df_cols = [*['ns', 'Ts', 'V0s'], *Is_cols]
    synth_data=pd.DataFrame(data,columns=df_cols)
    #print(synth_data)
    if geometry=='cylinder':
        synth_data.to_csv('synth_data_cyl.csv')
    elif geometry=='mNLP':
        synth_data.to_csv('synth_data_mNLP.csv')
    return synth_data
