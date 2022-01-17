import numpy as np
import matplotlib.pyplot as plt
import langmuir as l
from tqdm import tqdm
from itertools import count
from localreg import RBFnet, plot_corr
from localreg.metrics import rms_rel_error
path='plots/'

l1=15e-3
l2=40e-3
l3=15e-3
l4=40e-3
r0=0.3e-3

geo1 = l.Cylinder(r=r0, l=l1, lguard=float('inf'))
geo2 = l.Cylinder(r=r0, l=l2, lguard=float('inf'))
geo3 = l.Cylinder(r=r0, l=l3, lguard=float('inf'))
geo4 = l.Cylinder(r=r0, l=l4, lguard=float('inf'))

model = l.finite_length_current
Vs_geo1 = np.array([5]) # bias voltages
Vs_geo2 = np.array([4]) # bias voltages
Vs_geo3 = np.array([3]) # bias voltages
Vs_geo4 = np.array([1])  # bias voltages
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

N = 1000
ns  = rand_log(N, [4e10, 3e11])  # densities
###Ts = rand_uniform(N, [100, 2700]) # temperature
Ts = rand_uniform(N, [800, 3000]) # temperatures
V0s = rand_uniform(N, [-3, 0])   # floating potentials

# Generate probe currents corresponding to plasma parameters
Is_geo1 = np.zeros((N,len(Vs_geo1)))
Is_geo2 = np.zeros((N,len(Vs_geo2)))
Is_geo3 = np.zeros((N,len(Vs_geo3)))
Is_geo4 = np.zeros((N,len(Vs_geo4)))
for i, n, T, V0 in zip(count(), ns, Ts, tqdm(V0s)):
    Is_geo1[i] = model(geo1, l.Electron(n=n, T=T), V=V0+Vs_geo1)
    Is_geo2[i] = model(geo2, l.Electron(n=n, T=T), V=V0+Vs_geo2)
    Is_geo3[i] = model(geo3, l.Electron(n=n, T=T), V=V0+Vs_geo3)
    Is_geo4[i] = model(geo4, l.Electron(n=n, T=T), V=V0+Vs_geo4)
Is=np.append(Is_geo1,Is_geo2,axis=1)
Is=np.append(Is,Is_geo3,axis=1)
Is=np.append(Is,Is_geo4,axis=1)
"""
PART 2: TRAIN AND TEST THE REGRESSION NETWORK
"""

# Use M first data points for training, the rest for testing.
M = int(0.8*N)


# Train by minimizing relative error in temperature
net = RBFnet()
net.train(Is[:M], Ts[:M], num=250, relative=True, measure=rms_rel_error)


# Plot and print error metrics on test data
pred = net.predict(Is[M:])
Vs_geo1_str=np.array2string(Vs_geo1, formatter={'float_kind':lambda x: "%.1f" % x})
Vs_geo2_str=np.array2string(Vs_geo2, formatter={'float_kind':lambda x: "%.1f" % x})
Vs_geo3_str=np.array2string(Vs_geo3, formatter={'float_kind':lambda x: "%.1f" % x})
Vs_geo4_str=np.array2string(Vs_geo4, formatter={'float_kind':lambda x: "%.1f" % x})
fig, ax = plt.subplots()
#plot_corr(ax, ns[M:], pred, log=True)
plot_corr(ax, Ts[M:], pred, log=True)
plt.savefig(path+'2_'+'%.3f_'%l1+'%.3f_'%l2+'%.3f_'%l3+'%.3f_'%l4+'%s_'%Vs_geo1_str+'%s'%Vs_geo2_str+'%s_'%Vs_geo3_str+'%s_'%Vs_geo4_str+'_corr.png', bbox_inches="tight")
#print("RMS of relative density error: {:.1f}%".format(100*rms_rel_error(ns[M:], pred)))
print("RMS of relative density error: {:.1f}%".format(100*rms_rel_error(Ts[M:], pred)))

"""
PART 3: PREDICT PLASMA PARAMETERS FROM ACTUAL DATA
"""

data = l.generate_synthetic_data(geo1, Vs_geo1, model=model)
I_geo1 = np.zeros((len( data['ne']),len(Vs_geo1)))
I_geo2 = np.zeros((len( data['ne']),len(Vs_geo2)))
I_geo3 = np.zeros((len( data['ne']),len(Vs_geo3)))
I_geo4 = np.zeros((len( data['ne']),len(Vs_geo4)))
for i, n, T, V0 in zip(count(), data['ne'], data['Te'], tqdm(data['V0'])):
    I_geo1[i] = model(geo1, l.Electron(n=n, T=T), V=V0+Vs_geo1)
    I_geo2[i] = model(geo2, l.Electron(n=n, T=T), V=V0+Vs_geo2)
    I_geo3[i] = model(geo3, l.Electron(n=n, T=T), V=V0+Vs_geo3)
    I_geo4[i] = model(geo4, l.Electron(n=n, T=T), V=V0+Vs_geo4)
I=np.append(I_geo1,I_geo2,axis=1)
I=np.append(I,I_geo3,axis=1)
I=np.append(I,I_geo4,axis=1)

pred = net.predict(I)


plt.figure()

plt.plot(data['Te'], data['alt'], label='Ground truth')
plt.plot(pred, data['alt'], label='Predicted')
plt.xlabel('Temperature $[\mathrm{K}]$')
plt.ylabel('Altitude $[\mathrm{km}]$')

textstr=('l1=%.3f'%l1+' l2=%.3f'%l2+' l3=%.3f'%l3+' l4=%.3f'%l4+' V1=%s'%Vs_geo1_str+' V2=%s'%Vs_geo2_str+' V3=%s'%Vs_geo3_str+' V4=%s'%Vs_geo4_str)
#textstr(Vs_geo1[i] for i in range(0,len(Vs_geo1)))
plt.text(-0.02, 1.01, textstr, fontsize=14, transform=plt.gcf().transFigure)
plt.legend()
plt.savefig(path+'2_'+'%.3f_'%l1+'%.3f_'%l2+'%.3f_'%l3+'%.3f_'%l4+'%s_'%Vs_geo1_str+'%s'%Vs_geo2_str+'%s_'%Vs_geo3_str+'%s_'%Vs_geo4_str+'_corr.png', bbox_inches="tight")
plt.show()