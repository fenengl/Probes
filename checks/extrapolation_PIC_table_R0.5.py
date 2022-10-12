from __future__ import division
from langmuir.tables import *
#from tables_v2 import*
from langmuir.geometry import *
from langmuir.species import *
from langmuir.misc import *
from scipy import interpolate
from scipy.interpolate import interpn
from scipy.constants import value as constants
from copy import deepcopy
import scipy.special as special
from scipy import optimize
import os
import matplotlib.pyplot as plt
import numpy as np
import langmuir as l
from scipy.optimize import curve_fit
from itertools import count
import scipy.constants as sc


rs=np.array([0.002440,0.002440,0.002588,0.002728,0.002861,0.002988,0.003110,0.003228,0.003341,0.003450 ])
R=0.5
Te=np.array([500.00,500.00,562.50,625.00,687.50,750.00,812.50,875.00,937.50,1000.00])
ne=1e11
V=np.array([1.08,1.29,1.94,2.69,3.55,4.52,5.60,6.79,8.08,9.48])

normalization=None
keyword = ' sphere'

q=sc.elementary_charge
m=sc.electron_mass
tol = 1e-6 # For float comparisons
k = constants('Boltzmann constant')
V = make_array(V)

I0 = np.zeros((len(rs)))
if normalization is None:
    for i in range(0,len(rs)):

        I0[i] = normalization_current(l.Sphere(r=rs[i]), l.Electron(n=ne, T=Te[i]))
        print(I0[i])
elif normalization.lower() == 'thmax':
    I0 = 1

else:
    raise ValueError('Normalization not supported: {}'.format(normalization))

def powerlaw( x, a, b, c):
    return a*(b+x)**c
def logarithmus( x, a, b):
    return a*np.log(b+x)
axtest=np.array([0, 0.2, 0.3, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20,30,40, 50,60,70,80,90, 100,110])
axsim=np.array([25,30,40, 50,60,70,80,90, 100,110])
ticks=np.array([0,10,20,30,40, 50,60,70,80,90, 100,110])

table = get_table('laframboise'+keyword)
ax = table['axes']
vals = table['values']
vals=np.array(vals)
I_etas = interpn((ax[0],ax[1],), vals,(R,ax[1]), method='linear')
popt, pcov = curve_fit(powerlaw, ax[1], I_etas)
I_OML = np.zeros((len(rs)))
for i in range(0,len(rs)):
    geometry=l.Sphere(r=rs[i])
    I_OML[i] = l.OML_current(geometry, l.Electron(n=ne,T=Te[i]), eta=axsim[i], normalization='th')
print(I_OML)

cmap = plt.get_cmap('plasma', len(ax[1]))
plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'xtick.major.size': 20})
plt.rcParams.update({'ytick.major.size': 20})
plt.rcParams.update({'xtick.minor.size': 15})
plt.rcParams.update({'ytick.minor.size': 15})
plt.rcParams.update({'lines.linewidth': 3})

fig, ax1 = plt.subplots(figsize=(10, 10))
plt.xlabel('$\eta$ ')
plt.ylabel('I [A]')


y_est = powerlaw(axtest, *popt)
y_err = powerlaw(axtest, *popt)*0.06

plt.scatter(ax[1],I_etas,marker='o',s=55,c='b',label='Laframboise')
plt.plot(axtest,y_est,c='b',label='a*(b+x)**c')
ax1.fill_between(axtest, y_est - y_err, y_est + y_err, alpha=0.2,label='$\pm$ 6 %')
plt.plot(axsim,I_OML,c='k',label='OML')

plt.scatter(25,-1.06e-06/I0[0],marker='o',s=55,c='red',label='PTetra')
plt.scatter(30,-1.26e-06/I0[1],marker='o',s=55,c='red')
plt.scatter(40,-1.99e-06/I0[2],marker='o',s=55,c='red')
plt.scatter(50,-2.9e-06/I0[3],marker='o',s=55,c='red')
plt.scatter(60,-4.01e-06/I0[4],marker='o',s=55,c='red')
plt.scatter(70,-5.24e-06/I0[5],marker='o',s=55,c='red')
plt.scatter(80,-6.6e-06/I0[6],marker='o',s=55,c='red')
plt.scatter(90,-8.36e-06/I0[7],marker='o',s=55,c='red')
plt.scatter(100,-1.02e-05/I0[8],marker='o',s=55,c='red')
plt.scatter(110,-1.22e-05/I0[9],marker='o',s=55,c='red')

ax1.set_xlim(-5,115)
ax1.set_ylim(-5,115)
ax1.set_xticks(ticks)
ax1.set_yticks(ticks)
plt.title('a)')

leg = plt.legend()

plt.tight_layout()

plt.savefig('I_eta_0.5.png', bbox_inches="tight")
plt.show()
