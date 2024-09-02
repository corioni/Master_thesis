import numpy as np
from matplotlib import pyplot as plt
import emcee
from astropy.table import Table
import pandas as pd
import corner
from ps_generator import PowerSpectrumBooster
import multiprocessing
from multiprocessing import Pool
import pickle

kmax = 1
kmin = 0.04
N_k = 50
redshift =0.1
k = np.geomspace(kmin,kmax, N_k)
delta_k = np.log(kmax/kmin)/N_k



def P(k,params):
    emulator_folder = './emulators_tuning_8/lightning_logs'
    pofkboostfunction_version = 1
    # Unpack parameters array
    As, Omm,  mu0 = params
#    As, ns, Omb, Omm, h, mnu, w, wa, mu0 = params
    # Create a dictionary for cosmological parameters
    cosmo_par = {
        'As': As,
        'ns': 0.96,
        'Omb':0.05,
        'Omm': Omm,
        'h': 0.68,#h,
        'mnu': 0.15,
        'w': -1.0,
        'wa': 0.0,
    }    
    redshifts = [redshift]
    booster = PowerSpectrumBooster(emulator_folder, pofkboostfunction_version)
    k, boosted_plin, boosted_pnl, boost_emulator = booster.boosted_power_spectrum(cosmo_par,mu0, redshifts,k)
    return boosted_pnl


def N(k):
    V =  256 **3
    N_k = k**3 * delta_k * V /(2* np.pi**2)  
    return N_k

def C(ki,kj,params):
    return (2/N(ki))*(P(ki,params))**2


def loglikelihood(params, observed_P, k):
    model_P = P(k, params)  
    variance = C(k, k, params)  
    if np.any(variance <= 0):
        return -np.inf
    diff = observed_P - model_P
    log_p = -0.5 * ((diff ** 2) / variance + np.log(2 * np.pi * variance))
    return np.sum(log_p)

def logprior(params):
    As, Omm, mu0 = params
#    As,  Omm,  mu0 = params   
    # Define the intervals for each parameter
    intervals = {
        'mu0':  [-0.1, 0.1],
        'Omm':  [0.24, 0.4],
        'h':    [0.61, 0.73],
        'A_s':  [1.7e-9, 2.5e-9],
        'w':    [-1.3, -0.7],
        'wa':   [-0.7, 0.5],
        'mnu':  [0.0, 0.15],
        'Omb':  [0.04, 0.06],
        'ns':   [0.92, 1.00],
    }

    # Check each parameter against its interval
    if (
        intervals['mu0'][0] <= mu0 <= intervals['mu0'][1] and
        intervals['Omm'][0] <= Omm <= intervals['Omm'][1] and
        #intervals['h'][0] <= h <= intervals['h'][1] and
        intervals['A_s'][0] <= As <= intervals['A_s'][1] 
        #intervals['w'][0] <= w <= intervals['w'][1] and
        #intervals['wa'][0] <= wa <= intervals['wa'][1] and
        #intervals['mnu'][0] <= mnu <= intervals['mnu'][1] and
        #intervals['Omb'][0] <= Omb <= intervals['Omb'][1] and
        #intervals['ns'][0] <= ns <= intervals['ns'][1]
    ):
        return 0
    else:
        return -np.inf

def logposterior(params, P,k):
    lp = logprior(params)
    if not np.isinf(lp):
        return lp+loglikelihood(params, P,k)
    else:
        return lp


#######################################
# PARAMETRI SCELTI
#######################################
cosmo_par = [ 2.1e-09, 0.32, -0.01]

yerr =  np.sqrt(C(k,k,cosmo_par))
Pnl = P(k,cosmo_par)
# Generate observed Pnl with errors added
Pnl_obs = Pnl + np.random.normal(0, yerr)

ndim = 3
nwalkers = 32


# Define the intervals for each parameter in the specified order
intervals = {
    'As':  [1.7e-9, 2.5e-9],
    #'ns':  [0.92, 1.00],
    #'Omb': [0.04, 0.06],
    'Omm': [0.24, 0.4],
    #'h':   [0.61, 0.73],
    #'mnu': [0.0, 0.15],
    #'w':   [-1.3, -0.7],
    #'wa':  [-0.7, 0.5],
    'mu0': [-0.1, 0.1],
}

# Function to generate a random parameter set
def generate_random_parameters(intervals):
 
    sorted_keys = ['As', 'Omm','mu0']
    return [np.random.uniform(*intervals[key]) for key in sorted_keys]

# Generate parameters 20 times
pos = np.array([generate_random_parameters(intervals) for _ in range(nwalkers)])
assert pos.shape == (nwalkers, len(intervals)), "Shape mismatch"


run_mcmc = True
if run_mcmc:
    # Initializing the sampler with the number of walkers, dimensions, and log-posterior
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logposterior, args=(Pnl_obs, k))

    # The 'state' variable now contains the state of the chain after the last sample

    sampler = emcee.EnsembleSampler(nwalkers,ndim,logposterior,args=(Pnl_obs,k));
    state = sampler.run_mcmc(pos, 50)
    sampler.reset()
    state = sampler.run_mcmc(state, 5000)


    # Saving the sampler to a file using pickle
    with open('sampler2_z=0.1_epochs=5000.pkl', 'wb') as f:
        pickle.dump(sampler, f)
