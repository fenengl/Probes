import langmuir as l
import numpy as np
import matplotlib.pyplot as plt
from finite_radius_extrapolated import *

# plasma = [ # Approximately F-region plasma
#     Electron(n=1e11, T=1000),
#     # Oxygen(n=1e11, T=1000)
#     ]

#l1=25e-3
#r0=0.255e-3
rs=10e-3#l.Electron(n=1e11, T=1600).debye*4.5###10e-3 ### ICI2 rocket parameters
rs2=20e-3
rs3=30e-3
rs4=1e-6
geometry = l.Sphere(r=rs)
geometry2 = l.Sphere(r=rs2)
geometry3 = l.Sphere(r=rs3)
geometry4 = l.Sphere(r=rs4)
#geometry = Sphere(r=3.33*electron.debye)
#geo2 = l.Cylinder(r=r0, l=l1, lguard=float('inf'))

#model = finite_length_current
#model_sphere=l.OML_current

electron =      Electron(n=1e11, T=1000)

V = np.linspace(-1, 2, 400)
eta = np.linspace(0, 200, 400)
#print(eta)

I_OML = l.OML_current(geometry, electron, eta=eta, normalization='th')
I_FR = finite_radius_current(geometry, electron, eta=eta, normalization='th')


I_OML2 = l.OML_current(geometry2, electron, eta=eta, normalization='th')
I_FR2 = finite_radius_current(geometry2, electron, eta=eta, normalization='th')


I_OML3 = l.OML_current(geometry3, electron, eta=eta, normalization='th')
I_FR3 = finite_radius_current(geometry3, electron, eta=eta, normalization='th')


I_OML4 = l.OML_current(geometry4, electron, eta=eta, normalization='th')
I_FR4 = finite_radius_current(geometry4, electron, eta=eta, normalization='th')


plt.plot(eta, I_OML, label='OML')
plt.plot(eta, I_FR, label='FR')

plt.plot(eta, I_OML2, label='OML2')
plt.plot(eta, I_FR2, label='FR2')

plt.plot(eta, I_OML3, label='OML3')
plt.plot(eta, I_FR3, label='FR3')

plt.plot(eta, I_OML4, label='OML4')
plt.plot(eta, I_FR4, label='FR4')

plt.xlabel('eta')
plt.ylabel('I/I_th')
plt.legend()
plt.show()
