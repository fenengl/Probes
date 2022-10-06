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
from network_TF_DNN import *
from network_RBF import *
from calc_beta import beta_calc
from scipy.stats.stats import pearsonr
#from keras.models import load_model
#from tensorflow.keras.layers import Normalization





"""
Geometry, Probes and bias voltages
"""
version=2
l1=25e-3
l2=40e-3
r0=0.255e-3

geo1 = l.Cylinder(r=r0, l=l1, lguard=float('inf'))
geo2 = l.Cylinder(r=r0, l=l2, lguard=float('inf'))

model1 = finite_length_current
model2 = finite_length_current

Vs_geo1 = np.array([4]) # bias voltages
Vs_geo2 = np.array([2.5,7.5])#,5.5]) # bias voltages

geometry='cylinder'
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
    synth_data=pd.read_csv('../3_cyl_short/synth_data_cyl_2.csv',index_col=0)
else:
    logger.error('Specify whether to create new data or use the existing set')

ns =np.array(synth_data.iloc[:,0])
Ts =np.array(synth_data.iloc[:,1])
V0s=np.array(synth_data.iloc[:,2])
Is =np.array(synth_data.iloc[:,3:])


"""
PART 2: TRAIN AND TEST THE REGRESSION / TensorFlow NETWORK

"""
### select ratio of training, validation and testing data
M = int(0.7*N)
K= int(0.8*N)
TF=False

if TF == True:
    results,history,net_model= tensorflow_network(Is,Ts,M,K)
    net_model.save('tf_model_%i'%version)
    pred=net_model.predict(Is[K:])

    ax = pd.DataFrame(data=history.history).plot(figsize=(15, 7))
    ax.grid()
    _ = ax.set(title="Training loss and accuracy", xlabel="Epochs")
    _ = ax.legend(["Training loss", "Trainig accuracy","Validation loss", "Validation accuracy"])
    plt.savefig('history_c.png', bbox_inches="tight")
    plt.show()
elif TF == False:

    net_model = keras.models.load_model('../3_cyl_short/tf_model_%i'%version)
    pred=net_model.predict(Is[K:])

    #pred,net_model= rbf_network(Is,Ts,M)
else:
    logger.error('Specify whether to use tensorflow or RBF')



"""
PART 3: PREDICT PLASMA PARAMETERS FROM ACTUAL DATA (IRI)
"""

range=450
Vs_all=np.concatenate((Vs_geo1,Vs_geo2))

sel_noise=1e-7


data_geo1 = l.generate_synthetic_data(geo1, Vs_geo1, model=model1,noise=sel_noise)
data_geo2 = l.generate_synthetic_data(geo2, Vs_geo2, model=model2,noise=sel_noise)

#I_geo1 = np.zeros((len( data['ne']),len(Vs_geo1)))
#I_geo2 = np.zeros((len( data['ne']),len(Vs_geo2)))

#for i, n, T, V0 in zip(count(), data['ne'], data['Te'], tqdm(data['V0'])):
#    I_geo1[i] = model1(geo1, l.Electron(n=n, T=T), V=V0+Vs_geo1)
#    I_geo2[i] = model2(geo2, l.Electron(n=n, T=T), V=V0+Vs_geo2)
I=np.append(data_geo1['I'],data_geo2['I'],axis=1)

print(data_geo1['Te'])
print(data_geo2['Te'])

predictions = net_model.predict(I)



"""
PART 4: ANALYSIS
"""
range1=120
range2=450
beta=beta_calc(l1,l2,r0,Vs_geo1,Vs_geo2,ns,Ts,V0s,Is)
#print(beta)
plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'xtick.major.size': 20})
plt.rcParams.update({'ytick.major.size': 20})
plt.rcParams.update({'xtick.minor.size': 15})
plt.rcParams.update({'ytick.minor.size': 15})
plt.rcParams.update({'lines.linewidth': 3})




fig, ax = plt.subplots(figsize=(10, 10))
plot = ax.plot
plot(data_geo2['Te'], data_geo2['alt'], label='Ground truth')
plot(predictions, data_geo2['alt'], label='Predicted')
#ax.set_aspect('equal', 'box')
ax.set_xlabel('Temperature $[\mathrm{K}]$')
ax.set_ylabel('Altitude $[\mathrm{km}]$')
ax.set_xlim(0,2800)
ax.set_ylim(75,525)



plt.text(40,420,'RMSRE ({0} - {1} km) = {2}%' .format(range1,range2,round(rms_rel_error(data_geo1['Te'].ravel()[range1:range2], predictions.ravel()[range1:range2])*100,1)))


plt.axhline(y=range2, color='red', linestyle='dotted', linewidth=3)
plt.axhline(y=range1, color='red', linestyle='dotted', linewidth=3)
ax.get_xaxis().set_major_formatter(mplot.ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(mplot.ticker.ScalarFormatter())
plt.title('a)')
plt.legend()
plt.savefig('predict_1.png', bbox_inches="tight")


print_table(
        [['l1'              , 'B1'              , 'V1'              ,'l2'              ,'B2'              , 'V2'              ,'l3'              ,'B3'               , 'V3'              ,'dB'              ],
        [l1,np.mean(beta.Beta_cyl1),Vs_geo1[0],l2,np.mean(beta.Beta_cyl2),Vs_geo2[0],l2,np.mean(beta.Beta_cyl3),Vs_geo2[1],np.mean(beta.diff_Beta)]])


print_table(
        [['sigma'              ,'RMSRE'              , 'corr'                  ],
        [sel_noise, rms_rel_error(data_geo1['Te'].ravel(), predictions.ravel()),pearsonr(data_geo1['Te'].ravel(),predictions.ravel())[0]]])
