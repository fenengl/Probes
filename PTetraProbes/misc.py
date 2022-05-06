#!/usr/bin/env python3
import numpy as np
from scipy.constants import value as constants

eta = np.array([10, 20, 30, 40 ,50 ,60, 70, 80, 90, 100])
kb = constants('Boltzmann constant')
Te = 600
e = constants('elementary charge')
v = (eta*Te*kb)/e

print(v)
