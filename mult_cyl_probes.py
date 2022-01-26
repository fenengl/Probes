import numpy as np
import matplotlib.pyplot as plt
import langmuir as l
from tqdm import tqdm
from itertools import count
from localreg import RBFnet, plot_corr
from localreg.metrics import rms_error, rms_rel_error
from frmt import print_table
import scipy.constants as sc

path='plots/'

l1=25e-3
l2=40e-3
r0=0.255e-3
geo1 = l.Cylinder(r=r0, l=l1, lguard=float('inf'))
geo2 = l.Cylinder(r=r0, l=l2, lguard=float('inf'))
model = l.finite_length_current
Vs_geo1 = np.array([2.5,7.5]) # bias voltages
Vs_geo2 = np.array([4,5.5]) # bias voltages


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


N = 1000
ns  = rand_log(N, [4e10, 3e11])  # densities
Ts = rand_uniform(N, [800, 3000]) # temperatures
V0s = rand_uniform(N, [-1,  0])   # floating potentials

Vs_all=np.concatenate((Vs_geo1,Vs_geo2))
Vmax=np.max([Vs_geo1,Vs_geo2])


cond=(calc_eta(Vmax,Ts)<95)
Ts,V0s,ns=Ts[cond],V0s[cond],ns[cond]
N=len(Ts)


# Generate probe currents corresponding to plasma parameters
Is_geo1 = np.zeros((N,len(Vs_geo1)))
Is_geo2 = np.zeros((N,len(Vs_geo2)))


for i, n, T, V0 in zip(count(), ns, Ts, tqdm(V0s)):
        Is_geo1[i] = model(geo1, l.Electron(n=n, T=T), V=V0+Vs_geo1)
        Is_geo2[i] = model(geo2, l.Electron(n=n, T=T), V=V0+Vs_geo2)
Is=np.append(Is_geo1,Is_geo2,axis=1)




"""
PART 2: TRAIN AND TEST THE REGRESSION NETWORK
"""

# Use M first data points for training, the rest for testing.
M = int(0.8*N)


# Train by minimizing relative error in temperature
net = RBFnet()
net.train(Is[:M], Ts[:M], num=200, relative=True, measure=rms_rel_error) ### 180/200 works okay/250

# Plot and print error metrics on test data
pred = net.predict(Is[M:])

Vs_geo1_str=np.array2string(Vs_geo1, formatter={'float_kind':lambda x: "%.1f" % x})
Vs_geo2_str=np.array2string(Vs_geo2, formatter={'float_kind':lambda x: "%.1f" % x})

fig, ax = plt.subplots()
#plot_corr(ax, ns[M:], pred, log=True)
plot_corr(ax, Ts[M:], pred, log=True)
#plt.savefig('4probes.png', bbox_inches="tight")
#print("RMS of relative density error: {:.1f}%".format(100*rms_rel_error(ns[M:], pred)))
print("RMS of relative density error: {:.1f}%".format(100*rms_rel_error(Ts[M:], pred)))
############################### save until here
"""
PART 3: PREDICT PLASMA PARAMETERS FROM ACTUAL DATA
"""

data = l.generate_synthetic_data(geo2, Vs_all, model=model,noise=0) ####3<<<<<<<<<<this is where the eta warningscome from

cond=(calc_eta(Vmax,data['Te'])<95)
data['Te'],data['V0'],data['ne'],data['alt']=data['Te'][cond],data['V0'][cond],data['ne'][cond],data['alt'][cond]


I_geo1 = np.zeros((len( data['ne']),len(Vs_geo1)))
I_geo2 = np.zeros((len( data['ne']),len(Vs_geo2)))

for i, n, T, V0 in zip(count(), data['ne'], data['Te'], tqdm(data['V0'])):
    I_geo1[i] = model(geo1, l.Electron(n=n, T=T), V=V0+Vs_geo1)
    I_geo2[i] = model(geo2, l.Electron(n=n, T=T), V=V0+Vs_geo2)
I=np.append(I_geo1,I_geo2,axis=1)

pred = net.predict(I)


plt.figure()

plt.plot(data['Te'], data['alt'], label='Ground truth')
plt.plot(pred, data['alt'], label='Predicted')
plt.xlabel('Temperature $[\mathrm{K}]$')
plt.ylabel('Altitude $[\mathrm{km}]$')
plt.legend()
plt.show()

#plt.figure()
#plt.plot(data['V0'],data['alt'])
#plt.xlabel('VO')
#plt.ylabel('Altitude $[\mathrm{km}]$')
#plt.show()

print_table(
    [['RMSE'              , 'RMSRE'                  ],
     [rms_error(data['Te'], pred), rms_rel_error(data['Te'] , pred)]])

all_var=np.array([ns,Ts,V0s])
all_var=np.append(all_var.T,Is,axis=1)
np.savetxt('Ne_Te_V0s_Is4_train.txt', all_var[:M])
np.savetxt('Ne_Te_V0s_Is4_test.txt', all_var[M:])





# eta = np.zeros((N,len(Vs_all)))
# for k,j in zip(count(), Vs_all): 
#     eta[:,k]=(sc.elementary_charge*(V0s+j))/(sc.Boltzmann*Ts)
# eta[eta>100]=np.nan
# #np.where(eta>100)
# cond=~np.isnan(eta).any(axis=1)
