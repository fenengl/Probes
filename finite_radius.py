from langmuir import *
import numpy as np
import matplotlib.pyplot as plt

# plasma = [ # Approximately F-region plasma
#     Electron(n=1e11, T=1000),
#     # Oxygen(n=1e11, T=1000)
#     ]
electron =      Electron(n=1e11, T=1000)

geometry = Sphere(r=3.33*electron.debye)

V = np.linspace(-1, 2, 100)
eta = np.linspace(0, 25, 100)

I_OML = OML_current(geometry, electron, eta=eta, normalization='th')
I_FR = finite_radius_current(geometry, electron, eta=eta, normalization='th')

plt.plot(eta, I_OML, label='OML')
plt.plot(eta, I_FR, label='FR')
plt.xlabel('eta')
plt.ylabel('I/I_th')
plt.legend()
plt.show()
