#!/usr/bin/env python3
"""
Plotting script for PTetra
Author: Sigvald Marholm, 2022
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import langmuir as l
from funcs import *
from argparse import ArgumentParser
from os.path import join as pjoin

parser = ArgumentParser(description="Plotting tool for PTetra workshop")
parser.add_argument('folder', help='Folder to PTetra simulation')
parser.add_argument('column', default='sc_i_tot', nargs='?', help='Which column of pictetra.hst to plot')
parser.add_argument('-r', metavar='tau', type=float, default=1e-7, help='Relaxation time')
parser.add_argument('--OML', action='store_true', help='Compare current with OML theory')
parser.add_argument('--FR', action='store_true', help='Compare current with finite-radius theory')
args = parser.parse_args()

df = read_hst(args.folder)
geometry, electron, voltages = parse_parameters(args.folder)

av = average(df, args.column, args.r)
last_value = av.iat[-1]

if args.OML:
    comparison_value = l.OML_current(geometry, electron, V=voltages[0])

if args.FR:
    comparison_value = l.finite_radius_current(geometry, electron, V=voltages[0])
    # comparison_value = l.finite_length_current(geometry, electron, V=voltages[0])

geometry_str = "cylinder" if isinstance(geometry, l.Cylinder) else "sphere"

print("Geometry:           {}".format(geometry_str))
print("Radius:             {:.2f} debye lengths".format(geometry.r/electron.debye))
if geometry_str=="cylinder":
        print("Length:             {:.1f} debye lengths".format(geometry.l/electron.debye))
print("Temperature:        {:.2f} K ({:.3g} eV)".format(electron.T, electron.eV))
print("Density:            {:.3g} /m^3".format(electron.n))
print("Debye length:       {:.3g} mm".format(electron.debye*1e3))
print()

print("Last plotted value: {:.3g}".format(last_value))

if args.OML or args.FR:
    print("Comparsion current: {:.3g}".format(comparison_value*1e6))
    print("Error:              {:.2f}%".format(100*(last_value-comparison_value)/comparison_value, "%"))
    plt.axhline(comparison_value, linestyle='--', color='k', label='Comparison')

plt.plot(df.time*1e6, df[args.column], '#ccc', zorder=-100, label='Raw')
plt.plot(df.time*1e6, av, label=r'Averaged ($\tau={:g}\,\mathrm{{\mu s}}$)'.format(args.r*1e6))
plt.xlabel('Time [us]')
plt.ylabel('Quantity: {}'.format(args.column))
# plt.ylabel('Current [A]')
plt.legend()
plt.savefig(pjoin(args.folder,args.column+'_.png'),dpi=300)
plt.show()
