#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import emcee
from astropy.table import Table
import pandas as pd
import corner
from ps_generator import PowerSpectrumBooster
import multiprocessing
from multiprocessing import Pool



kmax = 1
kmin = 0.04
N_k = 50
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
    redshifts = [0.1]
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
    # Assuming P(k, params) is the model prediction function for power spectrum
    # and 'observed_P' is the numpy array containing the observed power spectrum data.
    model_P = P(k, params)  # Now P(k, params) is clearly a function call
    variance = C(k, k, params)  # This function or value should return the variance at each k

    # Make sure variance is not zero to avoid division by zero
    if np.any(variance <= 0):
        return -np.inf

    diff = observed_P - model_P
    log_p = -0.5 * ((diff ** 2) / variance + np.log(2 * np.pi * variance))
    return np.sum(log_p)

def logprior(params):
#    As, ns, Omb, Omm, h, mnu, w, wa, mu0 = params
    As,  Omm,  mu0 = params   
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


# In[6]:


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
    # Sort the keys according to the specified order and then generate random parameters
    sorted_keys = ['As', 'Omm','mu0']
    #sorted_keys = ['As', 'ns', 'Omb', 'Omm', 'h', 'mnu', 'w', 'wa', 'mu0']
    return [np.random.uniform(*intervals[key]) for key in sorted_keys]

# Generate parameters 20 times
pos = np.array([generate_random_parameters(intervals) for _ in range(nwalkers)])

# Print the shape of the generated pos matrix
print("Shape of pos:", pos.shape)  
assert pos.shape == (nwalkers, len(intervals)), "Shape mismatch"






import pickle
# Load the sampler from the file
with open('sampler2_z=0.1_epochs=750.pkl', 'rb') as f:
    loaded_sampler = pickle.load(f)



import corner
import matplotlib.pyplot as plt

corner_figure = corner.corner(
    loaded_sampler.flatchain,
    quantiles=[0.16, 0.5, 0.84],
    #range=[(1.7e-9, 2.5e-9),(0.92, 1.00),(0.04, 0.06),(0.24, 0.4),(-0.11, 0.11)],
    #labels=[r'$A_s$', r'$n_s$', r'$\Omega_b$', r'$\Omega_{m}$', r'$h$', r'$\mu_0$'],
    labels=[r'$A_s$', r'$\Omega_{m}$',r'$\mu_0$'],
    show_titles=True,
    title_fmt='.2e',
    title_kwargs={"fontsize": 10},
    smooth1d =2.5,
    smooth = 2.5,
    # You can change the color of the histogram fill with the 'hist_kwargs' argument
    hist_kwargs={"color": "c", "linewidth": 1.5, "alpha": 0.9},#,"fill":True},
    # Or change the appearance of the density plots
    plot_density=False,#True,  # To disable density plot
    plot_datapoints=False,  # To plot the individual data points
    hist_bin_factor =5,
    max_n_ticks=5,
    data_kwargs={"marker": '.', "alpha": 0.5, "color": 'darkblue'},
    fill_contours=True,  # To fill the contour plots instead of just lines
    contourf_kwargs={"colors": None, "alpha": 0.5}  # Adjust contour colors and transparency
)

# The expected values
expected_values = {
    r'$A_s$': 2.1e-9,
#    r'$n_s$': 0.96,  
#    r'$\Omega_b$':0.05,
    r'$\Omega_{m}$': 0.32,
    r'$\mu_0$': -0.01
}
# Extract the axes from the corner plot to add lines
axes = np.array(corner_figure.axes).reshape((3, 3))

# Loop over the axes to add lines at the expected values
for i, param in enumerate([r'$A_s$',r'$\Omega_{m}$', r'$\mu_0$']):
    # Add vertical lines on each column
    for j in range(i +1):
        ax = axes[i,i ]
        ax.axvline(expected_values[param], color='r', linestyle='--') # For column parameters
    



# You might want to adjust the size of the figure
corner_figure.set_size_inches(8,8)

# Show or save the plot
plt.show()
# or
plt.savefig('corner_plot_750.pdf')


import pygtc

corner_data = loaded_sampler.flatchain

labels = [r'$A_s$', r'$\Omega_{m}$', r'$\mu_0$']
ranges = [(np.min(corner_data[:,0]), np.max(corner_data[:,0])),
          (np.min(corner_data[:,1]), np.max(corner_data[:,1])),
          (np.min(corner_data[:,2]), np.max(corner_data[:,2]))]
truths=((2.1e-9,0.32,-0.1),(2.1e-9,0.32,-0.1))

corner_figure = pygtc.plotGTC(
    corner_data,
    paramNames=labels,
    truths=truths,
    figureSize='MNRAS_page',
    plotName = 'fullGTC.pdf'
)




