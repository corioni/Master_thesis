import numpy as np
import json
import copy
import matplotlib
from smt.sampling_methods import LHS
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# =============================================
# Set plotting defaults
# =============================================
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams['text.usetex'] = True
params = {'xtick.top': True, 'ytick.right': True, 'xtick.direction': 'in', 'ytick.direction': 'in'}
plt.rcParams.update(params)


total_samples  = 30               # Number of samples to generate
prefix         = "Sim"         # A label for the simulations
dictfile       = "parameters_" + prefix + "_" + str(total_samples) + ".json" # Output dictionary file
test_run       = False# Don't make dict, but plot samples instead...

# Choose parameters to vary and the prior-range to vary
# We can look at e.g. the EuclidEmulator2 paper to pick the prior range for the emulator
parameters_to_vary = {
    'gravity_model_marta_mu0':  [-0.1, 0.1],
#    'Omega_cdm':    [0.2, 0.34],  # +-25%, Euclid has 20%
#    'h':            [0.60, 0.74],  # +-10% Euclid has 9%
#    'A_s':         [1.6e-9, 2.6e-9], # +-25%, Euclid has +-20%
}
parameters_to_vary_arr = list(parameters_to_vary.keys())  # More concise way to list keys

# Set the fiducial cosmology and simulations parameters
run_param_fiducial = {
    'label':        "FiducialCosmology",
    'outputfolder': "../FML/FML/COLASolver",
    'colaexe':      "../FML/FML/COLASolver/nbody",

    # COLA parameters
    'boxsize':    512.0,
    'Npart':      512,
    'Nmesh':      512,
    'Ntimesteps': 30,
    'Seed':       1234567,
    'zini':       20.0,
    'input_spectra_from_lcdm': "false",
    'sigma8_norm': "false",
  
    # Fiducial cosmological parameters - the ones we sample over will be changed below for each sample
    'cosmo_param': {
        'use_physical_parameters': False,
        'cosmology_model': 'LCDM',
        'gravity_model': 'Marta',
        'h':          0.67,
        'Omega_b':    0.049,
        'Omega_cdm':  0.27,
        'Omega_ncdm': 0.001387,
        'Omega_k':    0.0,
        'omega_fld':  0.0,
        'w0':         -1.0, 
        'wa':         0.0,
        'Neff':       3.046,
        'k_pivot':    0.05,
        'A_s':        2.1e-9,
        'sigma8':     0.83,
        'n_s':        0.96,
        'T_cmb':      2.7255,
        'log10fofr0': -5.0,
        'gravity_model_marta_mu0':  0.,
        'largscale_linear': 'false',
        'kmax_hmpc':  20.0,
    },

}

# Extract fiducial values
fiducial_values = [run_param_fiducial['cosmo_param'][param] for param in parameters_to_vary_arr]

# Generate all samples

############### LHS SAMPLING
#ranges = np.array([parameters_to_vary[key] for key in parameters_to_vary])
#sampling = LHS(xlimits=ranges)
#all_samples = sampling(total_samples)


################# UNIFORM SAMPLING
key = next(iter(parameters_to_vary))
range_ = parameters_to_vary[key]
all_samples = np.linspace(range_[0], range_[1], total_samples).reshape(-1, 1)

# Generate the dictionaries
simulations = {}
for count, sample in enumerate(all_samples):
    if test_run:
        print("===========================")
        print("New parameter sample:")
    run_param = copy.deepcopy(run_param_fiducial)
    for i, param in enumerate(parameters_to_vary):
        # change the values of the parameter to vary
        run_param["cosmo_param"][param] = sample[i]
        if test_run:
            print("Setting ", param, " to value ", sample[i])
    label = prefix + str(count)
    run_param['label'] = label

    simulations[str(count)] = copy.deepcopy(run_param)

if test_run:
    nparam = len(parameters_to_vary_arr)
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(nparam, nparam)
    labels = [r'$\mu_0$', r'$\Omega_\mathrm{cdm}$', r'$h$', r'$A_s$']

    for i in range(nparam):
        for j in range(nparam):
            if i > j:
                ax = fig.add_subplot(gs[i, j])
                ax.scatter(all_samples[:, j], all_samples[:, i], color ='#6699CC', s=5)
                ax.scatter(fiducial_values[j], fiducial_values[i], color='#f67d93', s=10, zorder=3)  # Fiducial value
                
                if i == nparam - 1:
                    ax.set_xlabel(labels[j])
                else:
                    ax.set_xticklabels([])

                if j == 0:
                    ax.set_ylabel(labels[i])
                else:
                    ax.set_yticklabels([])
            elif i == j:
                ax = fig.add_subplot(gs[i, j])
                ax.hist(all_samples[:, i], bins=30, color ='#6699CC', alpha=0.7)
                ax.axvline(fiducial_values[i], color='#f67d93', zorder=3)  # Fiducial value
                
                if i == nparam - 1:
                    ax.set_xlabel(labels[i])
                else:
                    ax.set_xticklabels([])

                if j == 0:
                    ax.set_ylabel('Frequency')
                else:
                    ax.set_yticklabels([])

    plt.tight_layout()
    plt.savefig("parameter_samples.pdf")
    plt.show()
    exit(1)

# Save to file
with open(dictfile, "w") as f:
    data = json.dumps(simulations, indent=4)
    f.write(data)
