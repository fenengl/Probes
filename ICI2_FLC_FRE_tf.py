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
import math
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from finite_radius_extrapolated import *
from finite_length_extrapolated import *
#from tensorflow.keras.layers import Normalization

l1=25e-3
r0=0.255e-3
rs=10e-3#l.Electron(n=1e11, T=1600).debye*4.5###10e-3 ### ICI2 rocket parameters

geo1 = l.Sphere(r=rs)
geo2 = l.Cylinder(r=r0, l=l1, lguard=float('inf'))

# a=l.Electron(n=1e11, T=1600).debye*1 ### *1 for cylinders
# print(a)
# breakpoint()
model = l.finite_length_current
model_sphere=finite_radius_current
#model_sphere2=l.OML_current

Vs_geo1 = np.array([4]) # bias voltages
Vs_geo2 = np.array([2.5,4,5.5,10]) # bias voltages last one was supposed to be 7- electronics issue caused it to be 10V

### works also with 2 cyl probes and a sphere

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
def calc_eta(V,T):
    eta=(sc.elementary_charge*V)/(sc.Boltzmann*T)
    return eta


N = 50000 #### has been increased from 5000 ---increase to 50000

ns  = rand_log(N, [4e10, 3e11])  # densities
#ns=np.repeat(1e11,N)
Ts = rand_uniform(N, [800, 3000]) # temperatures
#Ts=np.repeat(1600, N)
V0s = rand_uniform(N, [-1,  0])   # floating potentials
#V0s = np.linspace(-1, 0,num=N)

Vs_all=np.concatenate((Vs_geo1,Vs_geo2))
Vmax=np.max(Vs_all)

cond=(calc_eta(Vmax+V0s,Ts)<100)
Ts,V0s,ns=Ts[cond],V0s[cond],ns[cond]
N=len(Ts)



# plt.figure()
# plt.xlabel('eta')
# plt.ylabel('Ts')
# plt.scatter(Ts,calc_eta(V0s+Vs_geo1,Ts))
# plt.show()
# a=calc_eta(V0s+Vs_geo1,Ts)
# a=a[a<10]
# print(a)
# Generate probe currents corresponding to plasma parameters
Is_geo1 = np.zeros((N,len(Vs_geo1)))
Is_geo2 = np.zeros((N,len(Vs_geo2)))
Is_geo_OML = np.zeros((N,len(Vs_geo1)))

for i, n, T, V0 in zip(count(), ns, Ts, tqdm(V0s)):
        Is_geo1[i] = model_sphere(geo1, l.Electron(n=n, T=T),V=V0+Vs_geo1)
        #Is_geo_OML[i] = model_sphere2(geo1, l.Electron(n=n, T=T),V=V0+Vs_geo1)
        Is_geo2[i] = model(geo2, l.Electron(n=n, T=T), V=V0+Vs_geo2)
Is=np.append(Is_geo1,Is_geo2,axis=1)

# eta=calc_eta(V0s+Vs_geo1,Ts)
# plt.scatter(V0s+Vs_geo1, -Is_geo_OML*1e6, marker='.', label='OML')
# plt.scatter(V0s+Vs_geo1, -Is_geo1*1e6, marker='.',label='FR')
# plt.xlabel('V [V]')
# plt.ylabel('I [uA]')
# plt.legend()
# plt.show()
# breakpoint()


# Isat=sc.elementary_charge*4*sc.pi*ns*np.sqrt((sc.Boltzmann*Ts)/(2*sc.pi*sc.m_e))
#
# Isat=np.array(Isat)
# Is_1=np.array(Is[:,1])
# print(Isat.shape)
# print(Is_1.shape)

# allvar=np.column_stack((Isat,Is_1,Ts))
# np.savetxt('Isat_Is_Ts.txt', allvar)
# breakpoint()
"""
PART 2: TRAIN AND TEST THE REGRESSION NETWORK
"""

# Use M first data points for training, the rest for testing.
M = int(0.8*N)

#training_features = Is[:M]
#validation_features = Is[M:]
#training_labels = Ts[:M]
#validation_labels = Ts[M:]


## normalizer

normal = preprocessing.Normalization()
normal.adapt(Is[:M])
#a=normal(Is[:M])
#np.savetxt('norm.txt',a)
#layer = Normalization(axis=None)
#layer.adapt(Is)

input_size = Is[:M].shape[1]
#layers. Dense-- regular NN layer densely connected
tf_model = keras.Sequential([
 normal,
 layers.Dense(40, activation='relu', input_shape=[1]), #50
 layers.Dense(40, activation='relu'),#50
 layers.Dense(1, activation='relu')
])

# tf_model = keras.Sequential([
#  normal,
#  layers.Dense(200, activation='relu', input_shape=[1]), #50
#  layers.Dense(40, activation='relu'),
#  layers.Dense(40, activation='relu'),#50
#  layers.Dense(1, activation='relu')
# ])------ this worked down to 150



#tf_model = keras.Sequential([
# normalizer,
# layers.Dense(10000, activation='relu'),
 #layers.Dense(1000, activation='relu'),
 #layers.Dense(400, activation='relu'),
 #layers.Dense(10, activation='relu'),
 #layers.Dense(1)
#])
#tf_model = keras.Sequential([  worked well with n=5000
# normalizer,
 #layers.Dense(400, activation='relu'),
 #layers.Dense(10, activation='relu'),
 #layers.Dense(1)
#])
##https://towardsdatascience.com/why-rectified-linear-unit-relu-in-deep-learning-and-the-best-practice-to-use-it-with-tensorflow-e9880933b7ef


#tf_model.compile(optimizer='sgd', loss='mean_squared_error')
tf_model.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(0.001))#set 0.001#,metrics=['accuracy']) #remove acc ## works also with logarithmic, when increasing learning rate, finishes in less steps but less accurate
##https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
##mean_squared_error: Mathematically, it is the preferred loss function under the inference framework of maximum likelihood if the distribution of the target variable is Gaussian. It is the loss function to be evaluated first and only changed if you have a good reason.
##mean_squared_logaritmic_error: target value has a spread of values and when predicting a large value, you may not want to punish a model as heavily as mean squared error
##mean_absolute_error:  more robust to outliers,
### documentation to optimizers https://towardsdatascience.com/7-tips-to-choose-the-best-optimizer-47bb9c1219e adam good first try, also SGD later potentially
###https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
###https://keras.io/api/optimizers/adam/
###TensorFlow: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08. --> default values
###Keras: lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0.

with open('report.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    tf_model.summary(print_fn=lambda x: fh.write(x + '\n'))


## train
# Train by minimizing relative error in temperature
#net.train(Is[:M], Ts[:M], num=250, relative=True, measure=rms_rel_error)



#trained=model.fit(xs, ys, epochs=500)
history=tf_model.fit(Is[:M], Ts[:M],
               validation_data=(Is[M:], Ts[M:]), ## add this again
               verbose=1, epochs=60)#set 100

ax = pd.DataFrame(data=history.history).plot(figsize=(15, 7))
ax.grid()
_ = ax.set(title="Training loss and accuracy", xlabel="Epochs")
_ = ax.legend(["Training loss", "Trainig accuracy"])
##https://www.architecture-performance.fr/ap_blog/saving-a-tf-keras-model-with-data-normalization/

results = tf_model.evaluate(Is[M:], Ts[M:], verbose=1)

pred=tf_model.predict(Is[M:])

fig, ax = plt.subplots()
plot_corr(ax, Ts[M:], pred,log=True)
plt.title('NN correlation plot')
plt.savefig('ICI2.png', bbox_inches="tight")
#plt.show()
"""
PART 3: PREDICT PLASMA PARAMETERS FROM ACTUAL DATA
"""


data = l.generate_synthetic_data(geo1, Vs_geo1, model=model,noise=0)
cond=(calc_eta(Vmax+data['V0'],data['Te'])<100)
data['Te'],data['V0'],data['ne'],data['alt']=data['Te'][cond],data['V0'][cond],data['ne'][cond],data['alt'][cond]

# print(max(data['V0']), min(data['V0']))
# print(calc_eta(data['V0']+4,data['Te']))
I_geo1 = np.zeros((len( data['ne']),len(Vs_geo1)))
I_geo2 = np.zeros((len( data['ne']),len(Vs_geo2)))

for i, n, T, V0 in zip(count(), data['ne'], data['Te'], tqdm(data['V0'])):
    I_geo1[i] = model_sphere(geo1, l.Electron(n=n, T=T), V=V0+Vs_geo1)
    I_geo2[i] = model(geo2, l.Electron(n=n, T=T), V=V0+Vs_geo2)


I=np.append(I_geo1,I_geo2,axis=1)


pred_data = tf_model.predict(I)


plt.figure()
plt.plot(data['Te'], data['alt'], label='Ground truth')
plt.plot(pred_data, data['alt'], label='Predicted')
plt.xlabel('Temperature $[\mathrm{K}]$')
plt.ylabel('Altitude $[\mathrm{km}]$')

#print_table(
#    [['RMSE'              , 'RMSRE'                  ],
#     [rms_error(data['Te'], pred), rms_rel_error(data['Te'] , pred)]])

plt.legend()
plt.savefig('ICI2_predict.png', bbox_inches="tight")
plt.show()
