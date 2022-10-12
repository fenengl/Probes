from langmuir import *
import numpy as np
import matplotlib.pyplot as plt
from finite_radius_extrapolated import *

data=np.load('iri.npz')#
for key in data.keys():
    print(key)                        # x
    print(data[key])  
