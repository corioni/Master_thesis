import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# =============================================
# Set plotting defaults
# =============================================
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 9 })
matplotlib.rcParams['text.usetex'] = True
params = {'xtick.top': True, 'ytick.right': True, 'xtick.direction': 'in', 'ytick.direction': 'in'}
plt.rcParams.update(params)
colormap = cm.viridis

# Path to the directory containing your power spectra files
data_dir = '../output/Test_Param'


# Specify the number of spectra files

params = {
    'Lbox': [1024,512,256],
    'Npart': [256,512,768],
    'Fmesh': [256,512,768],
    'timesteps': [5,10,30,50]
}

nrows = len(params)

fig, axes = plt.subplots(nrows, 2, figsize=(8, 10))
axes = np.array(axes).reshape(-1, 2)  # Ensure axes is always a 2D array

for idx, (param, values) in enumerate(params.items()):
    ax1, ax2 = axes[idx]
    value_mean = np.mean(values)
    value_range = np.ptp(values)  # Range of the values (max - min)
    norm = mcolors.Normalize(vmin=value_mean * 0.1, vmax=value_mean * 1.8)

    for value in values:
        pofk_file = f'{data_dir}/pofk_Sim_Test_{param}_{value}_cb_z0.000.txt'
        massfunc_file = f'{data_dir}/snapshot_Sim_Test_{param}_{value}_z0.000/massfunc_z0.000.txt'
        
        if os.path.exists(pofk_file) and os.path.exists(massfunc_file):
            power_spectrum = np.loadtxt(pofk_file, skiprows=1)
            mass_function = np.loadtxt(massfunc_file, skiprows=1)
            
            print(f"Loaded {pofk_file}: {power_spectrum.shape}")
            print(f"Loaded {massfunc_file}: {mass_function.shape}")
        else:
            print(f"Missing data for {param} = {value}")

        power_spectrum = np.loadtxt(f'{data_dir}/pofk_Sim_Test_{param}_{value}_cb_z0.000.txt', skiprows=1)
        mass_function = np.loadtxt(f'{data_dir}/snapshot_Sim_Test_{param}_{value}_z0.000/massfunc_z0.000.txt', skiprows=1)
        color = colormap(norm(value))
        ax1.loglog(power_spectrum[:, 0], power_spectrum[:, 1], linewidth = 0.8, label=f'{param} = {value}', color=color)
        ax2.loglog(mass_function[:, 0], mass_function[:, 1], linewidth = 0.8, label=f'{param} = {value}', color=color)
        if param == 'Lbox':
            ax1.axvline(2 * np.pi * 1.25 / value, linestyle='--', linewidth = 0.8, alpha=0.7, color=color)
            ax1.axvline(np.pi * 512 / value, linestyle='dotted', linewidth = 0.8, alpha=0.7, color=color)
            ax1.loglog(power_spectrum[:, 0], power_spectrum[:, 3], linewidth = 0.8, alpha=0.4, color='orange')
        elif param in ['Npart', 'Fmesh']:
            ax1.axvline(2 * np.pi * 1.25 / 512, linestyle='--', alpha=0.4, linewidth = 0.8, color='orange')
            ax1.axvline(np.pi * value / 512, linestyle='dotted', alpha=0.7, linewidth = 0.8, color=color)
        elif param == 'timesteps':
            ax1.axvline(2 * np.pi * 1.25 / 512, linestyle='--', color='orange', linewidth = 0.8, alpha=0.4)
            ax1.axvline(np.pi * 512 / 512, linestyle='dotted', linewidth = 0.8, color=color)
    ax1.loglog(power_spectrum[:, 0], power_spectrum[:, 3],alpha=0.4, color='orange', linewidth = 0.8, label='EuclidEmulator2')
    
    # Set plot properties for the first subplot
    ax1.set_title(f'Power Spectrum, z=0.0') if idx==0 else None
    ax1.set_ylabel('$P(k)$')
    ax1.set_xlim(5e-3, 7)
    ax1.set_xticklabels([]) if idx < nrows-1 else None
    # Set plot properties for the second subplot
    ax2.set_title(f'Mass function, z=0.0') if idx==0 else None
    ax2.set_ylabel('$n(m)$')
    ax2.set_xlim(1e10, 1e16)
    ax2.set_xticklabels([]) if idx < nrows-1 else None     

    ax1.legend(loc='lower left')
    # ax1.grid(True)
    ax2.legend(loc='lower left')
    # ax2.grid(True)


axes[-1,0].set_xlabel('$m$ $[M_\odot/h]$')
axes[-1,1].set_xlabel('$k$ Mpc$/h$')

fig.subplots_adjust(hspace=0)
#plt.tight_layout()
plt.savefig(f'pictures/parameter_tuning_all.pdf',bbox_inches='tight')

plt.show()
