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
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from finite_length_extrapolated import *
from data_gen import *
from network_TF_ne import *
from network_RBF import *
#from tensorflow.keras.layers import Normalization

"""
Geometry, Probes and bias voltages
"""

l1=25e-3
l2=25e-3
r0=0.255e-3

geo1 = l.Cylinder(r=r0, l=l1, lguard=float('inf'))
geo2 = l.Cylinder(r=r0, l=l2, lguard=float('inf'))

model1 = finite_length_current
model2 = finite_length_current

Vs_geo1 = np.array([2.5,5.5]) # bias voltages
Vs_geo2 = np.array([4,10])#,5.5]) # bias voltages

geometry='cylinder'
####l.Electron(n=4e11, T=800).debye*0.2 ### *1 for cylinders

"""
PART 1: GENERATE SYNTHETIC DATA USING LANGMUIR

"""
gendata=0
N = 10000 ## how many data points

### adjust the data limits in the class
if gendata == 1:
    synth_data=random_synthetic_data(N,geo1,geo2,model1,model2,Vs_geo1,Vs_geo2,geometry)
elif gendata == 0:
    synth_data=pd.read_csv('synth_data_cyl.csv',index_col=0)
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
M = int(0.8*N)

TF=1

if TF == 1:
    pred,results,history,net_model= tensorflow_network(Is,V0s,M)
    ax = pd.DataFrame(data=history.history).plot(figsize=(15, 7))
    ax.grid()
    _ = ax.set(title="Training loss and accuracy", xlabel="Epochs")
    _ = ax.legend(["Training loss", "Trainig accuracy"])
elif TF == 0:
    pred,net_model= rbf_network(Is,ns,M)
else:
    logger.error('Specify whether to use tensorflow or RBF')


"""
PART 3: PREDICT PLASMA PARAMETERS FROM ACTUAL DATA (IRI)
"""
Vs_all=np.concatenate((Vs_geo1,Vs_geo2))

data = l.generate_synthetic_data(geo2, Vs_all, model=model1,noise=0)

I_geo1 = np.zeros((len( data['ne']),len(Vs_geo1)))
I_geo2 = np.zeros((len( data['ne']),len(Vs_geo2)))

for i, n, T, V0 in zip(count(), data['ne'], data['Te'], tqdm(data['V0'])):
    I_geo1[i] = model1(geo1, l.Electron(n=n, T=T), V=V0+Vs_geo1)
    I_geo2[i] = model2(geo2, l.Electron(n=n, T=T), V=V0+Vs_geo2)
I=np.append(I_geo1,I_geo2,axis=1)

pred = net_model.predict(I)

plt.figure()

plt.plot(data['V0'], data['alt'], label='Ground truth')
plt.plot(pred, data['alt'], label='Predicted')
plt.xlabel('density $[\mathrm{K}]$')
plt.ylabel('Altitude $[\mathrm{km}]$')
plt.legend()
plt.savefig('predict.png', bbox_inches="tight")
plt.show()


#print_table(
#    [['RMSE'              , 'RMSRE'                  ],
#     [rms_error(data['Te'], pred), rms_rel_error(data['Te'] , pred)]])
