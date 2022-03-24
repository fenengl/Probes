"""
Copyright 2018
    Sigvald Marholm <marholm@marebakken.com>
    Diako Darian <diako.darian@gmail.com>

This file is part of langmuir.

langmuir is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

langmuir is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with langmuir.  If not, see <http://www.gnu.org/licenses/>.
"""


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
from scipy.optimize import curve_fit

def finite_radius_current(geometry, species, V=None, eta=None, normalization=None,
                          table=None):#'laframboise-darian-marholm'):###'labramboise'):####
    """
    A current model taking into account the effects of finite radius by
    interpolating between tabulated normalized currents. The model only
    accounts for the attracted-species currents (for which eta<0). It does
    not extrapolate, but returns ``nan`` when the input parameters are outside
    the domain of the model. This happens when the normalized potential for any
    given species is less than -25, when kappa is less than 4, when alpha is
    more than 0.2 or when the radius is more than 10 or sometimes all the way
    up towards 100 (as the distribution approaches Maxwellian). Normally finite
    radius effects are negligible for radii less than 0.2 Debye lengths (spheres)
    or 1.0 Debye lengths (cylinders).

    The model can be based on the following tables, as decided by the ``table``
    parameter:

    - ``'laframboise'``.
      The de-facto standard tables for finite radius currents, tables 5c
      and 6c in Laframboise, "Theory of Spherical and Cylindrical Langmuir
      Probes in a Collisionless, Maxwellian Plasma at Rest", PhD Thesis.
      Covers Maxwellian plasmas only, probe radii ranging from 0 to 100 Debye
      lengths.

    - ``'darian-marholm uncomplete'``.
      These tables covers Maxwellian, Kappa, Cairns and Kappa-Cairns
      distributions for radii ranging from 0.2 Debye lengths (spheres) or
      1.0 Debye length (cylinders) up to 10 Debye lengths. They are not as
      accurate as ``'laframboise'`` for pure the Maxwellian, but usually within
      one or two percent.

    - ``'darian-marholm'``.
      Same as above, but this is complemented by adding analytical values from
      OML theory, thereby extending the range of valid radii down to zero Debye
      lengths. In addition, the values for zero potential are replaced by
      analytical values (i.e. the thermal current), since these are amongst the
      most inaccurate in the above, and more accurate values can be analytically
      computed.

    - ``'laframboise-darian-marholm'``.
      This replaces the tabulated values for the Maxwellian distribution in
      ``'darian-marholm'`` with those of Laframboise. Accordingly this table
      produces the most accurate result available in any situation, and has the
      widest available parameter domain, with the probe radius gradually
      increasing from 10 towards 100 Debye lengths as the distribution
      approaches the Maxwellian.

    Parameters
    ----------
    geometry: Plane, Cylinder or Sphere
        Probe geometry

    species: Species or array-like of Species
        Species constituting the background plasma

    V: float or array-like of floats
        Probe voltage(s) in [V]. Overrides eta.

    eta: float or array-like of floats
        Probe voltage(s) normalized by k*T/q, where q and T are the species'
        charge and temperature and k is Boltzmann's constant.

    normalization: 'th', 'thmax', 'oml', None
        Wether to normalize the output current by, respectively, the thermal
        current, the Maxwellian thermal current, the OML current, or not at
        all, i.e., current in [A/m].

    table: string
        Which table to use for interpolation. See detailed description above.

    Returns
    -------
    float if voltage is float. array of floats corresponding to voltage if
    voltage is array-like.
    """
    if isinstance(species, list):
        if normalization is not None:
            logger.error('Cannot normalize current to more than one species')
            return None
        if eta is not None:
            logger.error('Cannot normalize voltage to more than one species')
            return None
        I = 0
        for s in species:
            I += finite_radius_current(geometry, s, V, eta, table=table)
        return I

    q, m, n, T = species.q, species.m, species.n, species.T
    kappa, alpha = species.kappa, species.alpha

    tol = 1e-6 # For float comparisons

    k = constants('Boltzmann constant')

    if V is not None:
        V = make_array(V)
        eta = -q*V/(k*T)
    else:
        eta = make_array(eta)

    eta = deepcopy(eta)

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

    if isinstance(geometry, Sphere):
        keyword = ' sphere'
    elif isinstance(geometry, Cylinder):
        keyword = ' cylinder'
    else:
       raise ValueError('Geometry not supported: {}'.format(geometry))

    R = geometry.r/species.debye

    def powerlaw( x, a, b, c):
        return a*(b+x)**c
    #
    # axtest=np.array([0, 0.2, 0.3, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 50, 100])
    # #alpha=0.2
    # print(alpha,tol,1/kappa)

    if (alpha <= tol) and (1/kappa <= tol):
        table = get_table('laframboise'+keyword)
        ax = table['axes']
        vals = table['values']
        vals=np.array(vals)
        I_etas = interpn((ax[0],ax[1],), vals,(R,ax[1]), method='linear')
        popt, pcov = curve_fit(powerlaw, ax[1], I_etas)#,bounds=(0, [1, 1, 0.5]))
        # print(popt)
        I[indices_p]=I0*powerlaw(eta[indices_p], *popt)

        # cmap = plt.get_cmap('plasma', len(ax[1]))
        # plt.figure(1)
        # plt.xlabel('Rs')
        # plt.ylabel('Is')
        # for i in range(0,len(vals)):
        #     a=ax[1][i]
        #     plt.plot(ax[0],vals[:,i],c=cmap(i),label='eta = %.1f' %a)
        # plt.vlines(R,0,25,colors='k',linewidth=0.5,label='R = {0:.2f}'.format(R))
        # plt.scatter(R*np.ones_like(I_etas),I_etas,marker='.',c='r')
        # leg = plt.legend(bbox_to_anchor =(1.3, 1))
        # plt.tight_layout()
        # plt.show()
        #
        # plt.figure(2)
        # plt.xlabel('etas')
        # plt.ylabel('I_eta')
        # plt.scatter(ax[1],I_etas,marker='.',c='k',label='etas from laframboise')
        # #plt.plot(ax[1],powerlaw(ax[1], *popt),linewidth=0.5,c='b')
        # plt.plot(axtest,powerlaw(axtest, *popt),linewidth=0.5,c='b',label='a*(b+x)**c')
        # plt.scatter(eta[indices_p],powerlaw(eta[indices_p], *popt),marker='.',c='r',label='eta =%.1f' %eta[indices_p])
        # leg = plt.legend()
        # plt.tight_layout()
        # plt.show()
        # breakpoint()
        #if(kappa != float('inf') or alpha != 0):
        #    logger.warning("Using pure Laframboise tables discards spectral indices kappa and alpha")
    else:

        table = get_table('darian-marholm'+keyword)
        ax = table['axes']
        vals = table['values']
        vals=np.array(vals)
        vals = table['values']
        #print(ax)
        vals=np.array(vals)
        I_etas = interpn((ax[0],ax[1],ax[2],ax[3],), vals,(1/kappa, alpha, R, ax[3]), method='linear')
        popt, pcov = curve_fit(powerlaw, ax[3], I_etas)
        I[indices_p]=I0*powerlaw(eta[indices_p], *popt)
        # print(popt)
        # plt.figure(2)
        # plt.xlabel('etas')
        # plt.ylabel('I_eta')
        # plt.scatter(ax[3],I_etas,marker='.',c='k',label='etas from laframboise')
        # #plt.plot(ax[1],powerlaw(ax[1], *popt),linewidth=0.5,c='b')
        # plt.plot(axtest,powerlaw(axtest, *popt),linewidth=0.5,c='b',label='a*(b+x)**c')
        # plt.scatter(eta[indices_p],powerlaw(eta[indices_p], *popt),marker='.',c='r',label='eta =%.1f' %eta[indices_p])
        # leg = plt.legend()
        # plt.tight_layout()
        # plt.show()
        # breakpoint()""



    I[indices_n] = I0*OML_current(geometry, species, eta=eta[indices_n], normalization='thmax')

    if any(np.isnan(I)):
        logger.warning("Data points occurred outside the domain of tabulated values resulting in nan")

    return I[0] if len(I) == 1 else I
