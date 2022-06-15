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




rs=10e-3
#T=600
T=600
n=1e11
V=3.5

normalization=None
geometry = l.Sphere(r=rs)
species=l.Electron(n=n, T=T)
keyword = ' sphere'

q, m, n, T = species.q, species.m, species.n, species.T
kappa, alpha = species.kappa, species.alpha
tol = 1e-6 # For float comparisons

k = constants('Boltzmann constant')

V = make_array(V)
eta = -q*V/(k*T)
#Temp= -q*V/(k*20)
#print(Temp)
I = np.zeros_like(eta)

indices_n = np.where(eta < 0)[0]   # indices for repelled particles
indices_p = np.where(eta >= 0)[0]  # indices for attracted particles

if normalization is None:
    I0 = normalization_current(geometry, species)
elif normalization.lower() == 'thmax':
    I0 = 1
elif normalization.lower() == 'th':
    I0 = normalization_current(geometry, species)/\
         thermal_current(geometry, species)
elif normalization.lower() == 'oml':
    I0 = normalization_current(geometry, species)/\
         OML_current(geometry, species, eta=eta)
else:
    raise ValueError('Normalization not supported: {}'.format(normalization))

R = geometry.r/species.debye
print(geometry.r)
print(R)
def powerlaw( x, a, b, c):
    return a*(b+x)**c
def logarithmus( x, a, b):
    return a*np.log(b+x)
axtest=np.array([0, 0.2, 0.3, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20,30,40, 50,60,70,80,90, 100])
    # #alpha=0.2
    # print(alpha,tol,1/kappa)
ax_PIC_eta=np.array([10,20,30,40,50,60,70,80,90,100])
ax_PIC_I=np.array([-6.56e-06,-1.13e-05,-1.58e-05,-1.95e-05,-2.21e-05,-2.41e-05,-2.58e-05,-2.7e-05,-2.78e-05,-2.85e-05])/I0


if (alpha <= tol) and (1/kappa <= tol):
    table = get_table('laframboise'+keyword)
    ax = table['axes']
    vals = table['values']
    vals=np.array(vals)
    I_etas = interpn((ax[0],ax[1],), vals,(R,ax[1]), method='linear')
    etas_ext=np.hstack((ax[1],ax_PIC_eta))
    ind=np.argsort(etas_ext)
    etas_ext=etas_ext[ind]
    Is_ext=np.hstack((I_etas,ax_PIC_I))
    Is_ext=Is_ext[ind]
    ext=np.vstack((etas_ext,Is_ext))
    print(ext)
    print(ext[0,15:])
    print(ext[1,:])

    #popt, pcov = curve_fit(powerlaw, ax[1], I_etas)#, bounds=(0, [1, 1, 0.83]))
    popt, pcov = curve_fit(powerlaw, ext[0,15:], ext[1,15:])#, bounds=(0, [1, 1, 0.83]))
    # print(etas_ext[9:])
    I[indices_p]=I0*powerlaw(eta[indices_p], *popt)        ########
    I_OML = I0*l.OML_current(geometry, l.Electron(n=n,T=T), eta=axtest, normalization='th')



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

    plt.scatter(eta[indices_p],-2.55E-05,marker='o',s=50,c='green',label='PTetra')### for T=1500 K
    plt.scatter(eta[indices_p],-2.53e-05,marker='o',s=15,c='orange',label='PUNC++')
    plt.scatter(10,-6.56e-06,marker='o',s=15,c='green')
    plt.scatter(20,-1.13e-05,marker='o',s=15,c='green')
    plt.scatter(30,-1.58e-05,marker='o',s=15,c='green')
    plt.scatter(40,-1.95e-05,marker='o',s=15,c='green')
    plt.scatter(50,-2.21e-05,marker='o',s=15,c='green')
    plt.scatter(60,-2.41e-05,marker='o',s=15,c='green')
    plt.scatter(70,-2.58e-05,marker='o',s=15,c='green')
    plt.scatter(80,-2.7e-05,marker='o',s=15,c='green')
    plt.scatter(90,-2.78e-05,marker='o',s=15,c='green')
    plt.scatter(100,-2.85e-05,marker='o',s=15,c='green')


    plt.scatter(ax[1],I0*I_etas,marker='.',c='k',label='etas from laframboise')
    #plt.scatter(67.7,-1.288656E-05/I0,marker='o',c='orange',label='PTetra')

        #plt.plot(ax[1],powerlaw(ax[1], *popt),linewidth=0.5,c='b')
    plt.plot(axtest,I0*powerlaw(axtest, *popt),linewidth=0.5,c='b',label='a*(b+x)**c')
    plt.plot(axtest,I_OML,linewidth=0.5,c='g',label='OML')
    plt.scatter(eta[indices_p],I[indices_p],marker='.',c='r',label='eta =%.1f' %eta[indices_p])
    leg = plt.legend()

    plt.tight_layout()
    plt.show()
    breakpoint()



    I[indices_n] = I0*OML_current(geometry, species, eta=eta[indices_n], normalization='thmax')
