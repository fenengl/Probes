import numpy as np
import matplotlib.pyplot as plt
import langmuir as l
from frmt import print_table
import scipy.constants as sc
import pandas as pd
import matplotlib.pyplot as plt
from itertools import count
from tqdm import tqdm
    ##  I= C*Ith(1+eta)^beta

def beta_calc(l1,l2,r0,V1,V2,ns,Ts,V0s,Is):

    C_cyl1=2*sc.pi*r0*l1*2/np.sqrt(sc.pi)
    C_cyl2=2*sc.pi*r0*l2*2/np.sqrt(sc.pi)
    #C_sph=4*sc.pi*rs**2
    #C_sph2=4*sc.pi*rs2**2

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


#    print(Is)
    Is1 =np.array(Is[:,0])
    Is2 =np.array(Is[:,1])
    Is3 =np.array(Is[:,2])

#    print(Is1)
    Beta_cyl1 = np.zeros(len(ns))
    Beta_cyl2 = np.zeros(len(ns))
    Beta_cyl3 = np.zeros(len(ns))
    diff_Beta = np.zeros(len(ns))
#    print(V2[0])
    for i, n, T, V0, I1 , I2, I3 in zip(count(), ns, Ts, tqdm(V0s),Is1,Is2,Is3):
        Cyl1=calc_neq(n)*calc_therm(T)*C_cyl1
        Cyl2=calc_neq(n)*calc_therm(T)*C_cyl2
        Beta_cyl1[i]=np.log(-I1/Cyl1)/np.log(1+calc_eta(V0+V1,T))
        Beta_cyl2[i]=np.log(-I2/Cyl2)/np.log(1+calc_eta(V0+V2[0],T))
        Beta_cyl3[i]=np.log(-I3/Cyl2)/np.log(1+calc_eta(V0+V2[1],T))
        diff_Beta[i]=Beta_cyl1[i]-Beta_cyl3[i]

    data=np.array([Beta_cyl1,Beta_cyl2,Beta_cyl3,diff_Beta]).T
    #print(data)
    df_cols = ['Beta_cyl1', 'Beta_cyl2', 'Beta_cyl3', 'diff_Beta']
    Beta=pd.DataFrame(data,columns=df_cols)

    return(Beta)
    #Beta.to_csv('Beta.csv')
    #print(Beta_sph)
    #print(Beta_sph2)
#print(Beta_cyl)
#print(Beta_cyl2)

#print(np.mean(diff_Beta))
#print(np.amin(diff_Beta))
#print(np.amax(diff_Beta))

    #print('T=',T)
    #print('V0=',V0)
    #print('n=',n)
    #print('Vs_cyl=',Vs_cyl)
    #print('Vs_sph=',Vs_sph)
    #Sph=calc_neq(n)*calc_therm(T)*C_sph
    #Sph2=calc_neq(n)*calc_therm(T)*C_sph2

    #print('etacyl=',calc_eta(V0+Vs_cyl,T))
    #print('etasph=',calc_eta(V0+Vs_sph,T))
    #print('Cyl=',Cyl)
    #print('Sph=',Sph)
    #print('log=',np.log(1+calc_eta(V0+Vs_sph,T)))
    #print('I1=',-I1)
    #print('I2=',-I2)
    #print('debye=',l.Electron(n=n, T=1).debye)
