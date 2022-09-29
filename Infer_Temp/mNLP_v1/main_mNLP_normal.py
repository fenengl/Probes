import numpy as np
import matplotlib.pyplot as plt
import langmuir as l
from tqdm import tqdm
from itertools import count
from localreg import RBFnet, plot_corr
from localreg.metrics import rms_error, rms_rel_error
from frmt import print_table
import scipy.constants as sc
import tensorflow as tf
import pandas as pd
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from finite_length_extrapolated import *
from finite_radius_extrapolated import *
from data_gen import *
from network_TF_DNN import *
from network_RBF import *
#from tensorflow.keras.layers import Normalization
version=5
"""
Geometry, Probes and bias voltages
"""

l1=25e-3###25
r0=0.255e-3
rs=10e-3#l.Electron(n=1e11, T=1600).debye*4.5###10e-3 ### ICI2 rocket parameters##10cd..

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
gendata=0
N = 50000 ## how many data points

### adjust the data limits in the class
if gendata == 1:
    synth_data=random_synthetic_data(N,geo1,geo2,model1,model2,Vs_geo1,Vs_geo2,geometry,version)
elif gendata == 0:
    synth_data=pd.read_csv('synth_data_mNLP_5.csv',index_col=0)
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
TF=1

if TF == 1:
    results,history,net_model= tensorflow_network(Is,Ts,M,K)
    ax = pd.DataFrame(data=history.history).plot(figsize=(15, 7))
    ax.grid()
    _ = ax.set(title="Training loss and accuracy", xlabel="Epochs")
    _ = ax.legend(["Training loss", "Trainig accuracy"])
elif TF == 0:
    pred,net_model= rbf_network(Is,Ts,M)
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


pred = net_model.predict(I)

plt.figure()

plt.plot(data['Te'], data['alt'], label='Ground truth')
plt.plot(pred, data['alt'], label='Predicted')
plt.xlabel('Temperature $[\mathrm{K}]$')
plt.ylabel('Altitude $[\mathrm{km}]$')
plt.legend()
plt.savefig('predict.png', bbox_inches="tight")
plt.show()

maxTe=data['Te'].max()
minTe=data['Te'].min()


maxNe=data['ne'].max()
minNe=data['ne'].min()

debyemax=l.Electron(n=maxNe, T=maxTe).debye
debyemin=l.Electron(n=minNe, T=minTe).debye
debyeminmax=l.Electron(n=minNe, T=maxTe).debye
debyemaxmin=l.Electron(n=maxNe, T=minTe).debye
print(debyemax)
print(debyemin)
print(debyeminmax)
print(debyemaxmin)

print(maxNe)
print(minNe)
print(maxTe)
print(minTe)

debye = np.zeros(len(data['Te']))


for i, nd, Td in zip(count(),data['ne'], data['Te']):
    debye[i]=l.Electron(n=nd, T=Td).debye

plt.figure()
plt.plot(debye, data['alt'], label='Debye')
plt.xlabel('Debye')
plt.ylabel('Altitude $[\mathrm{km}]$')
plt.legend()
plt.savefig('debye.png', bbox_inches="tight")
plt.show()

print(10e-3)
