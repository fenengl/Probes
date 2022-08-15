#!/usr/bin/env python3
import numpy as np
from langmuir import *
from scipy.constants import value as constants
import sys
import argparse

## Input Output Arguments ###
def main():
    ## Setup Params: #######


    eta = np.array([25])
    ne = np.ones(len(eta))*1e11
    teK = np.linspace(500,1000,len(eta)) #np.ones(len(eta))*600
    te = teK/11604.525
    RpLD = np.array([0.2, 0.3, 0.5, 1.0, 1.87])

    ## Formatted for automate_queue
    formatted = False

    mE      = constants('electron mass')
    eps0    = constants('electric constant')
    Q       = constants('elementary charge')
    kb = constants('Boltzmann constant')


    def etac(phi,te):
        return (Q*phi)/(kb*te)
    def probeb(eta,te):
        return (eta*kb*te)/(Q)

    omegaPe = np.zeros(ne.shape)
    DL_e = np.zeros(ne.shape)
    # eta = np.zeros(ne.shape)
    phi = np.zeros(ne.shape)
    Rp = np.zeros((len(eta),len(RpLD)))
    for i in range(len(eta)):
        omegaPe[i]    = Electron(n=ne[i], eV=te[i]).omega_p
        DL_e[i]   = Electron(n=ne[i], eV=te[i]).debye
        # eta[i] = etac(phi[i],teK[i])
        phi[i] = probeb(eta[i],teK[i])


        # print('Eta = %2.2f'%eta[i]+',  DL = %f'%DL_e[i]+' m, omegaPe = %2.2e'%omegaPe[i]+' 1/sec %2.2e'%(omegaPe[i]/2*np.pi)+' Hz, Phi = %2.2f'%phi[i])

    # for j in range(len(RpLD)):
    #     for i in range(len(eta)):
    #         Rp[i,j] = RpLD[j]*DL_e[i]
    #         print('Eta = %2.2f'%eta[i]+',  DL = %f'%DL_e[i]+' m, Te = %2.4f'%te[i]+'eV, Te = %2.4f'%teK[i]+'K,  Phi = %2.2f'%phi[i]+' V, Rp = %f'%Rp[i,j]+' m, Rp = %2.2f'%(Rp[i,j]/DL_e[i])+' LD')

    if formatted:
        print("ETA")
        for j in range(len(RpLD)):
            for i in range(len(eta)):
                sys.stdout.write('%2.2f '%eta[i])
                sys.stdout.flush()
        print("\nDL_e")
        for j in range(len(RpLD)):
            for i in range(len(eta)):
                sys.stdout.write('%f '%DL_e[i])
                sys.stdout.flush()
        print("\nTe_eV")
        for j in range(len(RpLD)):
            for i in range(len(eta)):
                sys.stdout.write('%2.4f '%te[i])
                sys.stdout.flush()
        print("\nTe_K")
        for j in range(len(RpLD)):
            for i in range(len(eta)):
                sys.stdout.write('%2.2f '%teK[i])
                sys.stdout.flush()
        print("\nphi")
        for j in range(len(RpLD)):
            for i in range(len(eta)):
                sys.stdout.write('%2.2f '%phi[i])
                sys.stdout.flush()
        print("\nRp_m")
        for j in range(len(RpLD)):
            for i in range(len(eta)):
                Rp[i,j] = RpLD[j]*DL_e[i]
                sys.stdout.write('%f '%Rp[i,j])
                sys.stdout.flush()
        print("\nRp_LD")
        for j in range(len(RpLD)):
            for i in range(len(eta)):
                Rp[i,j] = RpLD[j]*DL_e[i]
                sys.stdout.write('%2.2f '%(Rp[i,j]/DL_e[i]))
                sys.stdout.flush()
        print("")
    else:
        k = 0
        for j in range(len(RpLD)):
            for i in range(len(eta)):
                Rp[i,j] = RpLD[j]*DL_e[i]
                projectdir="Sphere_%2.2f"%(Rp[i,j]/DL_e[i])+"R_%2.2f"%phi[i]+"V_%2.2f"%phi[i]+"V_%2.2f"%teK[i]+"K"
                # print("Project Directory: "+projectdir)
                print('Run = %d'%k+', Project Directory: '+projectdir+', Eta = %2.2f'%eta[i]+',  DL = %f'%DL_e[i]+' m, Te = %2.4f'%te[i]+'eV, Te = %2.2f'%teK[i]+'K,  Phi = %2.2f'%phi[i]+' V, Rp = %f'%Rp[i,j]+' m, Rp = %2.2f'%(Rp[i,j]/DL_e[i])+' LD')

                k+=1


if __name__ == "__main__":
    main()
