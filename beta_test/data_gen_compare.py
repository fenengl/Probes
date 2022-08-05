import langmuir as l
import numpy as np
import scipy.constants as sc
import pandas as pd
from itertools import count
from tqdm import tqdm
from finite_length_extrapolated import *
from finite_radius_extrapolated import *
"""
PART 1: GENERATE SYNTHETIC DATA USING LANGMUIR
"""
def rand_log(N, range):
    """Generate N logarithmically distributed numbers in range"""
    x = rand_uniform(N, np.log(range))
    return np.exp(x)
def rand_uniform(N, range):
    """Generate N uniformly distributed numbers in range"""
    return range[0]+(range[1]-range[0])*np.random.rand(N)

l1=25e-3
r0=0.255e-3
rs=20e-3
l2=25e-3
rs2=10e-3

geo1 = l.Sphere(r=rs)
geo2 = l.Sphere(r=rs2)
geo3 = l.Cylinder(r=r0, l=l1, lguard=float('inf'))
geo4 =l.Cylinder(r=r0, l=l2, lguard=float('inf'))

model1 = finite_radius_current
model2 = finite_length_current

Vs_geo1 = np.array([4]) # bias voltages
Vs_geo2 = np.array([4]) # bias voltages last one was supposed to be 7- electronics issue caused it to be 10V

geometry='mNLP'
####l.Electron(n=4e11, T=800).debye*0.2 ### *1 for cylinders

"""
PART 1: GENERATE SYNTHETIC DATA USING LANGMUIR

"""

N = 5000 ## how many data points

ns  =rand_log(N, [4e10, 3e11]) # densities
Ts = rand_uniform(N, [600, 3000]) # temperatures### or 800
V0s =rand_uniform(N, [-1,  0]) # floating potentials


    # Generate probe currents corresponding to plasma parameters
Is_geo1 = np.zeros((N,len(Vs_geo1)))
Is_geo2 = np.zeros((N,len(Vs_geo2)))
Is_geo3 = np.zeros((N,len(Vs_geo2)))
Is_geo4 = np.zeros((N,len(Vs_geo2)))

for i, n, T, V0 in zip(count(), ns, Ts, tqdm(V0s)):
    Is_geo1[i] = model1(geo1, l.Electron(n=n, T=T),V=V0+Vs_geo1)
    Is_geo2[i] = model1(geo2, l.Electron(n=n, T=T), V=V0+Vs_geo1)
    Is_geo3[i] = model2(geo3, l.Electron(n=n, T=T), V=V0+Vs_geo2)
    Is_geo4[i] = model2(geo4, l.Electron(n=n, T=T), V=V0+Vs_geo2)

Is_temp=np.append(Is_geo1,Is_geo2,axis=1)
Is_temp2=np.append(Is_temp,Is_geo3,axis=1)
Is=np.append(Is_temp2,Is_geo4,axis=1)
#print(Is)
Is_cols = ["Is_{0}".format(x) for x in range(Is.shape[1])]
data=np.append(np.array([ns,Ts,V0s]).T,Is,axis=1)
df_cols = [*['ns', 'Ts', 'V0s'], *Is_cols]
synth_data=pd.DataFrame(data,columns=df_cols)
    #print(synth_data)
#print(V0s)
synth_data.to_csv('Beta_mNLP.csv')
