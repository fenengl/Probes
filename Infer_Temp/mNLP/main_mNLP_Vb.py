import numpy as np
import matplotlib.pyplot as plt
import langmuir as l
from tqdm import tqdm
from itertools import count
from localreg import RBFnet, plot_corr
from localreg.metrics import rms_error, rms_rel_error, error_std, max_abs_error, max_rel_error
from frmt import print_table
import scipy.constants as sc
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.constants as sc
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import sys
sys.path.append("..")
from finite_length_extrapolated import *
from data_gen import *
from network_TF_Vb import *
from network_RBF import *
from calc_beta import beta_calc_mNLP
from scipy.stats.stats import pearsonr
#from tensorflow.keras.layers import Normalization
version=6
"""
Geometry, Probes and bias voltages
"""

l1=25e-3###25
r0=0.255e-3
rs=3e-3#l.Electron(n=1e11, T=1600).debye*4.5###10e-3 ### ICI2 rocket parameters##10cd..

geo1 = l.Sphere(r=rs)
geo2 = l.Cylinder(r=r0, l=l1, lguard=float('inf'))

model1 = finite_radius_current
model2 = finite_length_current

Vs_geo1 = np.array([4]) # bias voltages
Vs_geo2 = np.array([2.5,4,5.5,10]) # bias voltages last one was supposed to be 7- electronics issue caused it to be 10V

geometry='mNLP'
####l.Electron(n=4e11, T=800).debye*0.2 ### *1 for cylinders

"""
PART 1: GENERATE SYNTHETIC DATA USING LANGMUIR

"""
gendata=False
N = 13000 ## how many data points

### adjust the data limits in the class
if gendata == True:
    synth_data=random_synthetic_data(N,geo1,geo2,model1,model2,Vs_geo1,Vs_geo2,geometry,version)
elif gendata == False:
    synth_data=pd.read_csv('synth_data_mNLP_2.csv',index_col=0)
else:
    logger.error('Specify whether to create new data or use the existing set')

ns =np.array(synth_data.iloc[:,0])
Ts =np.array(synth_data.iloc[:,1])
V0s=np.array(synth_data.iloc[:,2])
Is =np.array(synth_data.iloc[:,3:])

"""
PART 2: TRAIN AND TEST THE REGRESSION / TensorFlow NETWORK

"""
### select ratio of training and testing data

M = int(0.7*N)
K= int(0.8*N)
TF=False

if TF == True:
    results,history,net_model= tensorflow_network(Is,V0s,M,K)
    net_model.save('tf_model_%i'%version)
    pred=net_model.predict(Is[K:])

    ax = pd.DataFrame(data=history.history).plot(figsize=(15, 7))
    ax.grid()
    _ = ax.set(title="Training loss and accuracy", xlabel="Epochs")
    _ = ax.legend(["Training loss", "Trainig accuracy","Validation loss", "Validation accuracy"])
    plt.savefig('history_%i.png'%version, bbox_inches="tight")
    plt.show()
elif TF == False:

    net_model = keras.models.load_model('tf_model_%i'%version)
    pred=net_model.predict(Is[K:])

    #pred,net_model= rbf_network(Is,Ts,M)

else:
    logger.error('Specify whether to use tensorflow or RBF')


"""
PART 3: PREDICT PLASMA PARAMETERS FROM ACTUAL DATA (IRI)
"""

Vs_all=np.concatenate((Vs_geo1,Vs_geo2))



data = l.generate_synthetic_data(geo2, Vs_all, model=model2,noise=0)


I_geo1 = np.zeros((len( data['ne']),len(Vs_geo1)))
I_geo2 = np.zeros((len( data['ne']),len(Vs_geo2)))
 #####3 here is the problem:
for i, n, T, V0 in zip(count(), data['ne'], data['Te'], tqdm(data['V0'])):
    I_geo1[i] = model1(geo1, l.Electron(n=n, T=T), V=V0+Vs_geo1)
    I_geo2[i] = model2(geo2, l.Electron(n=n, T=T), V=V0+Vs_geo2)
I=np.append(I_geo1,I_geo2,axis=1)


predictions = net_model.predict(I)


"""
PART 4: ANALYSIS
"""

sel_noise=0
range1=120
range2=450
beta=beta_calc_mNLP(l1,r0,rs,Vs_geo1,Vs_geo2,ns,Ts,V0s,Is)
#print(beta)
plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'xtick.major.size': 20})
plt.rcParams.update({'ytick.major.size': 20})
plt.rcParams.update({'xtick.minor.size': 15})
plt.rcParams.update({'ytick.minor.size': 15})
plt.rcParams.update({'lines.linewidth': 3})


fig, ax = plt.subplots(figsize=(10, 10))
plot = ax.plot
plot(V0s[K:], pred, '+', ms=7)
xmin = min([min(V0s[K:]), min(pred)])
xmax = max([max(V0s[K:]), max(pred)])
plot([xmin, xmax], [xmin, xmax], '--k')
ax.set_aspect('equal', 'box')
ax.set_xlabel('synthetic $V_f [V]$')
ax.set_ylabel('predicted $V_f [V]$')
#ax.set_xticks([250,1000,3250])
#ax.set_yticks([250,1000,3250])
rmsre=rms_error(V0s[K:].ravel(),pred.ravel())
corrcoeff=pearsonr(V0s[K:].ravel(),pred.ravel())[0]

plt.text(-2,-0.25,'$l_n$={0} cm, $r_s$={1} cm\nRMSE = {2}' .format(l1*100,rs*100,round(rmsre,2)))
ax.get_xaxis().set_major_formatter(mplot.ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(mplot.ticker.ScalarFormatter())
plt.title('b)')
plt.savefig('correlation_%i.png'%version, bbox_inches="tight")

a=-pred.ravel()
b=-V0s[K:].ravel()
e = (a-b)/b
print(np.sum(np.abs(e)**2)/len(e))
print(np.max(a-b)**2)
print(np.sqrt(np.sum(np.abs(e)**2)/len(e)))
print(rmsre)
