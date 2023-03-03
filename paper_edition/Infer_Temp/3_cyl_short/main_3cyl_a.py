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
version=1
l1=25e-3
l2=30e-3
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
    synth_data=pd.read_csv('synth_data_cyl_%i.csv'%version,index_col=0)
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
    plt.savefig('history_%i.png'%version, bbox_inches="tight")
    plt.show()
elif TF == False:

    net_model = keras.models.load_model('tf_model_%i'%version)
    pred=net_model.predict(Is[K:])

    with open('rereport.txt','w') as fh:
        net_model.summary(print_fn=lambda x: fh.write(x + '\n'))
    #pred,net_model= rbf_network(Is,Ts,M)
else:
    logger.error('Specify whether to use tensorflow or RBF')


"""
PART 3: PREDICT PLASMA PARAMETERS FROM ACTUAL DATA (IRI)
"""


Vs_all=np.concatenate((Vs_geo1,Vs_geo2))

sel_noise=0
data = l.generate_synthetic_data(geo2, Vs_all, model=model1,noise=sel_noise)

I_geo1 = np.zeros((len( data['ne']),len(Vs_geo1)))
I_geo2 = np.zeros((len( data['ne']),len(Vs_geo2)))

for i, n, T, V0 in zip(count(), data['ne'], data['Te'], tqdm(data['V0'])):
    I_geo1[i] = model1(geo1, l.Electron(n=n, T=T), V=V0+Vs_geo1)
    I_geo2[i] = model2(geo2, l.Electron(n=n, T=T), V=V0+Vs_geo2)
I=np.append(I_geo1,I_geo2,axis=1)

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
plot = ax.loglog
plot(Ts[K:], pred, '+', ms=7)
xmin = min([min(Ts[K:]), min(pred)])
xmax = max([max(Ts[K:]), max(pred)])
plot([xmin, xmax], [xmin, xmax], '--k')
ax.set_aspect('equal', 'box')
ax.set_xlabel('synthetic $T_e$ [K]')
ax.set_ylabel('predicted $T_e$ [K]')
ax.set_xticks([250,1000,3250])
ax.set_yticks([250,1000,3250])
rmsre=rms_rel_error(Ts[K:].ravel(),pred.ravel())
corrcoeff=pearsonr(Ts[K:].ravel(),pred.ravel())[0]

plt.text(300,2000,'$l_1$={0} cm, $l_2$=$l_3$={1} cm\nRMSRE = {2}%' .format(l1*100,round(l2*100),round(rmsre*100,1)))
ax.get_xaxis().set_major_formatter(mplot.ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(mplot.ticker.ScalarFormatter())
plt.title('a)')
plt.savefig('correlation_%i.png'%version, bbox_inches="tight")



fig, ax = plt.subplots(figsize=(10, 10))
plot = ax.plot
plot(data['Te'], data['alt'], label='Ground truth IRI')
plot(predictions, data['alt'], label='Predicted')
#ax.set_aspect('equal', 'box')
ax.set_xlabel('Temperature $[\mathrm{K}]$')
ax.set_ylabel('Altitude $[\mathrm{km}]$')
ax.set_xlim(0,2800)
ax.set_ylim(75,525)


plt.text(40,420,'RMSRE ({0} - {1} km) = {2}%' .format(range1,range2,round(rms_rel_error(data['Te'].ravel()[range1:range2], predictions.ravel()[range1:range2])*100,1)))


plt.axhline(y=range2, color='red', linestyle='dotted', linewidth=3)
plt.axhline(y=range1, color='red', linestyle='dotted', linewidth=3)
ax.get_xaxis().set_major_formatter(mplot.ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(mplot.ticker.ScalarFormatter())
plt.title('b)')
plt.legend()
plt.savefig('predict_%i.png'%version, bbox_inches="tight")


print_table(
    [['l1'              , 'B1'              , 'B1_std'              , 'V1'              ,'l2'              ,'B2'              , 'B2_std'             , 'V2'              ,'l3'              ,'B3'               , 'B3_std'             , 'V3'              ,'dB'              ],
     [l1,np.mean(beta.Beta_cyl1),np.std(beta.Beta_cyl1),Vs_geo1[0],l2,np.mean(beta.Beta_cyl2),np.std(beta.Beta_cyl2),Vs_geo2[0],l2,np.mean(beta.Beta_cyl3),np.std(beta.Beta_cyl3),Vs_geo2[1],np.mean(beta.diff_Beta)]])


print_table(
        [['RMSE'             ,'RMSRE'             , 'corr'             ,'MAE'             ,'MRE'             ,'ER_STD'             ],
        [rms_error(Ts[K:].ravel(),pred.ravel()),rms_rel_error(Ts[K:].ravel(),pred.ravel()),pearsonr(Ts[K:].ravel(),pred.ravel())[0],max_abs_error(Ts[K:] ,pred.ravel()),max_rel_error(Ts[K:] ,pred.ravel()),error_std(Ts[K:].ravel(), pred.ravel())]])



print_table(
        [['sigma'              ,'RMSRE'              , 'corr'                  ],
        [sel_noise, rms_rel_error(data['Te'].ravel(), predictions.ravel()),pearsonr(data['Te'].ravel(),predictions.ravel())[0]]])

# debye = np.zeros(len(data['Te']))
# def calc_eta(V,T):
#     eta=(sc.elementary_charge*V)/(sc.Boltzmann*T)
#     return eta
#
# for i, nd, Td in zip(count(),data['ne'], data['Te']):
#     debye[i]=l.Electron(n=nd, T=Td).debye
#
# plt.figure()
# plt.plot(debye, data['alt'], label='Debye')
# plt.xlabel('Debye')
# plt.ylabel('Altitude $[\mathrm{km}]$')
# plt.legend()
#
# plt.savefig('debye_%i.png'%version, bbox_inches="tight")
#
#
# ratio=r0/debye
# plt.figure()
# plt.plot(ratio, data['alt'])
# plt.xlabel('ratio r0/debye')
# plt.ylabel('Altitude $[\mathrm{km}]$')
# plt.legend()
#
# plt.savefig('ratio_%i.png'%version, bbox_inches="tight")
#
#
# plt.figure()
# plt.plot(calc_eta(Vs_geo2[1],data['Te']), data['alt'])
# plt.xlabel('eta')
# plt.ylabel('Altitude $[\mathrm{km}]$')
# plt.legend()
#
# plt.savefig('eta_%i.png'%version, bbox_inches="tight")
#
