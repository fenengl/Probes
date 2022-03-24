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
#from tensorflow.keras.layers import Normalization

l1=25e-3
l2=40e-3
r0=0.255e-3
geo1 = l.Cylinder(r=r0, l=l1, lguard=float('inf'))
geo2 = l.Cylinder(r=r0, l=l2, lguard=float('inf'))
model = finite_length_current
Vs_geo1 = np.array([2.5,10]) # bias voltages
Vs_geo2 = np.array([4])#,5.5]) # bias voltages
### not possible for two  with different length but possible for 3 probes (2 lenghts)
l.Electron(n=4e11, T=800).debye*0.2 ### *1 for cylinders



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


N = 100000 #### has been increased from 5000
ns  = rand_log(N, [4e10, 3e11])  # densities
Ts = rand_uniform(N, [800, 3000]) # temperatures
V0s = rand_uniform(N, [-1,  0])   # floating potentials

Vs_all=np.concatenate((Vs_geo1,Vs_geo2))
Vmax=np.max(Vs_all)


cond=(calc_eta(Vmax,Ts)<90)
Ts,V0s,ns=Ts[cond],V0s[cond],ns[cond]
N=len(Ts)


# Generate probe currents corresponding to plasma parameters
Is_geo1 = np.zeros((N,len(Vs_geo1)))
Is_geo2 = np.zeros((N,len(Vs_geo2)))


for i, n, T, V0 in zip(count(), ns, Ts, tqdm(V0s)):
        Is_geo1[i] = model(geo1, l.Electron(n=n, T=T),V=V0+Vs_geo1)
        Is_geo2[i] = model(geo2, l.Electron(n=n, T=T), V=V0+Vs_geo2)
Is=np.append(Is_geo1,Is_geo2,axis=1)

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
tf_model.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(0.001))#,metrics=['accuracy']) #remove acc ## works also with logarithmic, when increasing learning rate, finishes in less steps but less accurate
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
               verbose=1, epochs=60)

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


data = l.generate_synthetic_data(geo1, Vs_geo1, model=model)#,noise=1)
cond=(calc_eta(Vmax,data['Te'])<90)
data['Te'],data['V0'],data['ne'],data['alt']=data['Te'][cond],data['V0'][cond],data['ne'][cond],data['alt'][cond]


I_geo1 = np.zeros((len( data['ne']),len(Vs_geo1)))
I_geo2 = np.zeros((len( data['ne']),len(Vs_geo2)))

for i, n, T, V0 in zip(count(), data['ne'], data['Te'], tqdm(data['V0'])):
    I_geo1[i] = model(geo1, l.Electron(n=n, T=T), V=V0+Vs_geo1)
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
