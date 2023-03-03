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
from calc_beta import beta_calc_sphere
from scipy.stats.stats import pearsonr
#from tensorflow.keras.layers import Normalization

"""
Geometry, Probes and bias voltages
"""
version=1
rs2=15e-3###25
rs=5e-3#l.Electron(n=1e11, T=1600).debye*4.5###10e-3 ### ICI2 rocket parameters##10cd..

geo1 = l.Sphere(r=rs)
geo2 = l.Sphere(r=rs2)

model1 = finite_radius_current
model2 = finite_radius_current

Vs_geo1 = np.array([4]) # bias voltages
Vs_geo2 = np.array([2.5,7.5]) # bias voltages last one was supposed to be 7- electronics issue caused it to be 10V

geometry='sphere'
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
    synth_data=pd.read_csv('synth_data_sphere_%i.csv'%version,index_col=0)
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
data = l.generate_synthetic_data(geo2, Vs_all, model=model2,noise=sel_noise)


I_geo1 = np.zeros((len( data['ne']),len(Vs_geo1)))
I_geo2 = np.zeros((len( data['ne']),len(Vs_geo2)))
 #####3 here is the problem:
for i, n, T, V0 in zip(count(), data['ne'], data['Te'], tqdm(data['V0'])):
    I_geo1[i] = model1(geo1, l.Electron(n=n, T=T), V=V0+Vs_geo1)
    I_geo2[i] = model2(geo2, l.Electron(n=n, T=T), V=V0+Vs_geo2)
I=np.append(I_geo1,I_geo2,axis=1)


predictions = net_model.predict(I)


I_geo1_test = np.zeros((len( data['ne']),len(Vs_geo1)))
I_geo2_test = np.zeros((len( data['ne']),len(Vs_geo2)))
 #####3 here is the problem:
for i, n, T, V0 in zip(count(), data['ne'], predictions, tqdm(data['V0'])):
    I_geo1_test[i] = model1(geo1, l.Electron(n=n, T=T), V=V0+Vs_geo1)
    I_geo2_test[i] = model2(geo2, l.Electron(n=n, T=T), V=V0+Vs_geo2)
I_test=np.append(I_geo1_test,I_geo2_test,axis=1)

"""
PART 4: ANALYSIS
"""
range1=120
range2=450
beta=beta_calc_sphere(rs,rs2,Vs_geo1,Vs_geo2,ns,Ts,V0s,Is)
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


plt.text(300,2000,'$r_1$={0} cm, $r_2$=$r_3$={1} cm\nRMSRE = {2}%' .format(rs*100,rs2*100,round(rmsre*100,1)))
ax.get_xaxis().set_major_formatter(mplot.ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(mplot.ticker.ScalarFormatter())
plt.title('a)')
plt.savefig('correlation_%i.png'%version, bbox_inches="tight")


Ix1=I[:,0]*1e6
Iy1=I_test[:,0]*1e6
Ix2=I[:,1]*1e6
Iy2=I_test[:,1]*1e6
Ix3=I[:,2]*1e6
Iy3=I_test[:,2]*1e6
fig, ax = plt.subplots(figsize=(10, 10))
plot = ax.plot
plot(Ix1, Iy1, 'c+', ms=14)
plot(Ix2, Iy2, 'c+', ms=14)
plot(Ix3, Iy3, 'c+', ms=14)
xmin = min([min(Ix1), min(Iy1),min(Ix2), min(Iy2),min(Ix3), min(Iy3)])
xmax = max([max(Ix1), max(Iy1),max(Ix2), max(Iy2),max(Ix3), max(Iy3)])
plot([xmin, xmax], [xmin, xmax], '--k')
ax.set_aspect('equal', 'box')
ax.set_xlabel('$I_s$ [μA]')
ax.set_ylabel('$I_i$ [μA]')
#ax.set_xticks([250,1000,3250])
#ax.set_yticks([250,1000,3250])

plt.text(-180,-10,'$r_1$={0} cm, $r_2$=$r_3$={1} cm' .format(rs*100,rs2*100))
plt.text(-180,-12,'RMSRE = {0}%' .format(rms_rel_error(Ix1, Iy1)*100))
ax.get_xaxis().set_major_formatter(mplot.ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(mplot.ticker.ScalarFormatter())
plt.title('b)')
plt.savefig('I_test_%i.png'%version, bbox_inches="tight")
plt.show()


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

#plt.xlim(0,2800)
#plt.ylim(50,550)
plt.axhline(y=range2, color='red', linestyle='dotted', linewidth=3)
plt.axhline(y=range1, color='red', linestyle='dotted', linewidth=3)
ax.get_xaxis().set_major_formatter(mplot.ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(mplot.ticker.ScalarFormatter())
plt.title('b)')
plt.legend()
plt.savefig('predict_%i.png'%version, bbox_inches="tight")



#print(l1)
#print(np.mean(beta.Beta_cyl1))
#print(np.mean(beta.diff_Beta))
#print(sel_noise)
#print(rms_rel_error(data['Te'][0:range], pred[0:range]))

#print(pearsonr(data['Te'].ravel(),pred.ravel())[0])
print_table(
    [['rs1'              , 'B1'              , 'V1'              ,'rs2'              ,'B2'              , 'V2'              ,'rs3'              ,'B3'               , 'V3'              ,'dB'             ],
     [rs,np.mean(beta.Beta_cyl1),Vs_geo1[0],rs2,np.mean(beta.Beta_cyl2),Vs_geo2[0],rs2,np.mean(beta.Beta_cyl3),Vs_geo2[1],np.mean(beta.diff_Beta)]])
print_table(
    [['B1_std'          ,'B2_std'           ,'B3_std'             ],
     [np.std(beta.Beta_cyl1),np.std(beta.Beta_cyl2),np.std(beta.Beta_cyl3)]])

print_table(
    [['RMSE'             ,'RMSRE'             , 'corr'             ,'MAE'             ,'MRE'             ,'ER_STD'             ],
     [rms_error(Ts[K:].ravel(),pred.ravel()),rms_rel_error(Ts[K:].ravel(),pred.ravel()),pearsonr(Ts[K:].ravel(),pred.ravel())[0],max_abs_error(Ts[K:] ,pred.ravel()),max_rel_error(Ts[K:] ,pred.ravel()),error_std(Ts[K:].ravel(), pred.ravel())]])

print_table(
    [['sigma'              ,'RMSRE'             , 'corr'                  ],
     [sel_noise, rms_rel_error(data['Te'].ravel(), predictions.ravel()),pearsonr(data['Te'].ravel(),predictions.ravel())[0]]])
