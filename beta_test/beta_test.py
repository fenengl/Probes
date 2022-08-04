

import numpy as np
import matplotlib.pyplot as plt
import langmuir as l
from frmt import print_table
import scipy.constants as sc
import pandas as pd
import matplotlib.pyplot as plt
from itertools import count
from tqdm import tqdm
## I= C*Ith(1+eta)^beta

l1=25e-3
r0=0.255e-3
rs=20e-3
l2=25e-3
rs2=10e-3


Vs_sph = np.array([4]) # bias voltages
Vs_cyl = np.array([4])

C_cyl=2*sc.pi*r0*l1*2/np.sqrt(sc.pi)
C_cyl2=2*sc.pi*r0*l2*2/np.sqrt(sc.pi)
C_sph=4*sc.pi*rs**2
C_sph2=4*sc.pi*rs2**2

def calc_eta(V,T):
    eta=(sc.elementary_charge*V)/(sc.Boltzmann*T)
    return eta
def calc_therm(T):
    therm=np.sqrt(sc.Boltzmann*T/(2*sc.pi*sc.m_e))
    return therm
def calc_neq(n):
    neq=n*sc.elementary_charge
    return neq
#eta = -q*V/(k*T)

synth_data=pd.read_csv('Beta_mNLP.csv',index_col=0)

ns =np.array(synth_data.iloc[:,0])
Ts =np.array(synth_data.iloc[:,1])
V0s=np.array(synth_data.iloc[:,2])
Is =np.array(synth_data.iloc[:,3])
Is2 =np.array(synth_data.iloc[:,4])
Ic =np.array(synth_data.iloc[:,5])
Ic2 =np.array(synth_data.iloc[:,6])

Beta_cyl = np.zeros(len(ns))
Beta_cyl2 = np.zeros(len(ns))
Beta_sph = np.zeros(len(ns))
Beta_sph2 = np.zeros(len(ns))
diff_Beta = np.zeros(len(ns))

for i, n, T, V0, I1 , I2 ,I3,I4 in zip(count(), ns, Ts, tqdm(V0s),Is,Is2,Ic,Ic2):
    Cyl=calc_neq(n)*calc_therm(T)*C_cyl
    Cyl2=calc_neq(n)*calc_therm(T)*C_cyl2
    #print('T=',T)
    #print('V0=',V0)
    #print('n=',n)
    #print('Vs_cyl=',Vs_cyl)
    #print('Vs_sph=',Vs_sph)
    Sph=calc_neq(n)*calc_therm(T)*C_sph
    Sph2=calc_neq(n)*calc_therm(T)*C_sph2

    #print('etacyl=',calc_eta(V0+Vs_cyl,T))
    #print('etasph=',calc_eta(V0+Vs_sph,T))
    #print('Cyl=',Cyl)
    #print('Sph=',Sph)
    #print('log=',np.log(1+calc_eta(V0+Vs_sph,T)))
    #print('I1=',-I1)
    #print('I2=',-I2)
    #print('debye=',l.Electron(n=n, T=1).debye)

    Beta_sph[i]=np.log(-I1/Sph)/np.log(1+calc_eta(V0+Vs_sph,T))
    Beta_sph2[i]=np.log(-I2/Sph2)/np.log(1+calc_eta(V0+Vs_sph,T))
    Beta_cyl[i]=np.log(-I3/Cyl)/np.log(1+calc_eta(V0+Vs_cyl,T))
    Beta_cyl2[i]=np.log(-I4/Cyl2)/np.log(1+calc_eta(V0+Vs_cyl,T))
    diff_Beta[i]=Beta_cyl[i]-Beta_sph[i]

data=np.array([Beta_cyl,Beta_sph,diff_Beta]).T
#print(data)
df_cols = ['Beta_cyl', 'Beta_sph', 'diff_Beta']
Beta=pd.DataFrame(data,columns=df_cols)
Beta.to_csv('Beta.csv')
print(Beta_sph)
print(Beta_sph2)
print(Beta_cyl)
print(Beta_cyl2)

print(np.mean(diff_Beta))
print(np.amin(diff_Beta))
print(np.amax(diff_Beta))
