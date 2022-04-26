"""
Useful functions for PTetra workshop
Author: Sigvald Marholm, 2022
"""

import pandas as pd
import numpy as np
import os
import re
import langmuir as l

def read_hst(folder_name):
    """ Read PTetra history file in folder """

    file_name = os.path.join(folder_name, 'pictetra.hst')

    column_names = [
        'timestep',
        'time',
        'netot',
        'nitot',
        'Te_eff',
        'pot1',
        'sc_phi_0',
        'sc_q_0',
        'sc_i_0',
        'sc_phi_1',
        'sc_q_1',
        'sc_i_1',
    ]

    df = pd.read_csv(file_name, sep='\s+', skiprows=2, names=column_names, index_col=0)

    # Add new column sc_i_tot for total collected spacecraft current
    df = df.assign(sc_i_tot=lambda a: a.sc_i_0 + a.sc_i_1)

    return df

def read_history(folder_name):
    """ Read PTetra history file in folder """

    file_name = os.path.join(folder_name, 'history.dat')

    column_names = [
    'n','time','ne','ni','KE','PE','V_0','I_0','Q_0','V_1','I_1','Q_1']

    df = pd.read_csv(file_name, sep='\s+', skiprows=4, names=column_names, index_col=0)

    # Add new column sc_i_tot for total collected spacecraft current
    df = df.assign(I_tot=lambda a: a.I_0 + a.I_1)

    return df

def parse_parameters(folder_name):
    """
    Parses geometry and plasma parameters from folder name and the pictetra.dat
    file inside it, and returns a geometry and Species object from Langmuir.

    Example:

        geometry, electron, voltages = parse_parameters('Sphere_0.5R_3V_3V')
        print(geometry)
        print(electron.debye)

    """

    dat_file = os.path.join(folder_name, 'pictetra.dat')

    with open(dat_file) as file:
        lines = file.read()

    # Read parameters from pictetra.dat
    density = float(re.search(r'\s+ne\s*=\s*(\S*)', lines).groups()[0])
    temperature = float(re.search(r'\s+te\s*=\s*(\S*)', lines).groups()[0])
    voltages = re.search(r'\$begin sc_fixedPot\s*(\S*)\s*(\S*)', lines)
    voltages = list(map(float, voltages.groups()))

    # Create Langmuir species
    electron = l.Electron(n=density, eV=temperature)
    # electron = l.Electron(n=density, eV=0.051704)
    # Read geometry from folder name
    radius = float(re.search(r'_([^_]*)R', folder_name).groups()[0])
    if 'cylinder' in folder_name.lower():
        length = float(re.search(r'_([^_]*)L', folder_name).groups()[0])
        geometry = l.Cylinder(r=radius*electron.debye, l=length*electron.debye)
    else: # sphere
        geometry = l.Sphere(r=radius*electron.debye)

    return geometry, electron, voltages

def average(df, column, relaxation_time):
    """
    Perform exponential moving average on a column in a Pandas dataframe. The
    relaxation time is in seconds.

    Example:

        df = read_hst('Sphere_0.5R_3V_3V')
        sc_i_tot_av = average(df, 'sc_i_tot', 1e-6)
    """
    delta_t = df.time[1]-df.time[0]
    alpha = 1-np.exp(-delta_t/relaxation_time)
    return df[column].ewm(alpha=alpha).mean()
