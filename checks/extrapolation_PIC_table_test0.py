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
#T=600
#print(rs)
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
#eta = -q*V/(k*T)

#Temp= -q*V/(k*20)
#print(Temp)
#I = np.zeros_like(eta)


I0 = np.zeros((len(rs)))
if normalization is None:
    for i in range(0,len(rs)):

        I0[i] = normalization_current(l.Sphere(r=rs[i]), l.Electron(n=ne, T=Te[i]))
        print(I0[i])
elif normalization.lower() == 'thmax':
    I0 = 1
#lif normalization.lower() == 'th':
#    I0 = normalization_current(geometry, species)/\
#         thermal_current(geometry, species)
#elif normalization.lower() == 'oml':
    #I0 = normalization_current(geometry, species)/\
    #     OML_current(geometry, species, eta=eta)
else:
    raise ValueError('Normalization not supported: {}'.format(normalization))

#R = geometry.r/species.debye
#print(geometry.r)
#print(R)
def powerlaw( x, a, b, c):
    return a*(b+x)**c
def logarithmus( x, a, b):
    return a*np.log(b+x)
axtest=np.array([0, 0.2, 0.3, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20,30,40, 50,60,70,80,90, 100,110])
axsim=np.array([25,30,40, 50,60,70,80,90, 100,110])
    # #alpha=0.2
    # print(alpha,tol,1/kappa)
#ax_PIC_eta=np.array([10,20,30,40,50,60,70,80,90,100])
#ax_PIC_I=np.array([-6.56e-06,-1.13e-05,-1.58e-05,-1.95e-05,-2.21e-05,-2.41e-05,-2.58e-05,-2.7e-05,-2.78e-05,-2.85e-05])/I0



table = get_table('laframboise'+keyword)
ax = table['axes']
vals = table['values']
vals=np.array(vals)
I_etas = interpn((ax[0],ax[1],), vals,(R,ax[1]), method='linear')
#etas_ext=np.hstack((ax[1],ax_PIC_eta))
#ind=np.argsort(etas_ext)
#etas_ext=etas_ext[ind]
#Is_ext=np.hstack((I_etas,ax_PIC_I))
#Is_ext=Is_ext[ind]
#ext=np.vstack((etas_ext,Is_ext))
#print(ext)
#print(ext[0,15:])
#print(ext[1,:])

popt, pcov = curve_fit(powerlaw, ax[1], I_etas)#, bounds=(0, [1, 1, 0.83]))
#popt, pcov = curve_fit(powerlaw, ext[0,15:], ext[1,15:])#, bounds=(0, [1, 1, 0.83]))
# print(etas_ext[9:])
#I[indices_p]=powerlaw(eta[indices_p], *popt)        ########
I_OML = np.zeros((len(rs)))
for i in range(0,len(rs)):
    geometry=l.Sphere(r=rs[i])
    I_OML[i] = l.OML_current(geometry, l.Electron(n=ne,T=Te[i]), eta=axsim[i], normalization='th')
print(I_OML)



cmap = plt.get_cmap('plasma', len(ax[1]))
# plt.figure(1)
# plt.xlabel('Rs')
# plt.ylabel('Is')
# for i in range(0,len(vals)):
#     a=ax[1][i]
#     plt.plot(ax[0],vals[:,i],c=cmap(i),label='eta = %.1f' %a)
#
# plt.vlines(R,0,25,colors='k',linewidth=0.5,label='R = {0:.2f}'.format(R))
# plt.scatter(R*np.ones_like(I_etas),I_etas,marker='.',c='r')
# leg = plt.legend(bbox_to_anchor =(1.3, 1))
# plt.tight_layout()
# plt.show()

plt.figure(2)
plt.xlabel('etas')
plt.ylabel('I')
#plt.scatter(eta[indices_p],-1.288656E-05,marker='o',c='orange') ### for T=600 K Last plotted value PTetra: -2.55e-05
#Last plotted value PUNC++: -2.53e-05


plt.scatter(25,-1.06e-06/I0[0],marker='o',s=15,c='green')
plt.scatter(30,-1.26e-06/I0[1],marker='o',s=15,c='green')
plt.scatter(40,-1.99e-06/I0[2],marker='o',s=15,c='green')
plt.scatter(50,-2.9e-06/I0[3],marker='o',s=15,c='green')
plt.scatter(60,-4.01e-06/I0[4],marker='o',s=15,c='green')
plt.scatter(70,-5.24e-06/I0[5],marker='o',s=15,c='green')
plt.scatter(80,-6.6e-06/I0[6],marker='o',s=15,c='green')
plt.scatter(90,-8.36e-06/I0[7],marker='o',s=15,c='green')
plt.scatter(100,-1.02e-05/I0[8],marker='o',s=15,c='green')
plt.scatter(110,-1.22e-05/I0[9],marker='o',s=15,c='green')

plt.scatter(ax[1],I_etas,marker='.',c='k',label='etas from laframboise')
#plt.scatter(67.7,-1.288656E-05/I0,marker='o',c='orange',label='PTetra')

    #plt.plot(ax[1],powerlaw(ax[1], *popt),linewidth=0.5,c='b')
plt.plot(axtest,powerlaw(axtest, *popt),linewidth=0.5,c='b',label='a*(b+x)**c')
plt.plot(axsim,I_OML,linewidth=0.5,c='g',label='OML')
#plt.scatter(eta[indices_p],I[indices_p],marker='.',c='r',label='eta =%.1f' %eta[indices_p])
leg = plt.legend()

plt.tight_layout()
plt.show()
#breakpoint()



#I[indices_n] = OML_current(geometry, species, eta=eta[indices_n], normalization='thmax')
