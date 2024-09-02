#!/usr/bin/env python
# coding: utf-8

# In[20]:


import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# =============================================
# Helper function to set plotting defaults
# =============================================
def set_plotting_defaults():
    try:
        matplotlib.rcParams['mathtext.fontset'] = 'cm'
        print("Set 'mathtext.fontset' to 'cm'")
    except Exception as e:
        print(f"Error setting 'mathtext.fontset': {e}")

    try:
        matplotlib.rcParams['font.family'] = 'STIXGeneral'
        print("Set 'font.family' to 'STIXGeneral'")
    except Exception as e:
        print(f"Error setting 'font.family': {e}")

    try:
        matplotlib.rcParams.update({'font.size': 18})
        print("Updated 'font.size' 18")
    except Exception as e:
        print(f"Error updating 'font.size': {e}")

    try:
        matplotlib.rcParams['text.usetex'] = True
        print("Enabled LaTeX text rendering with 'text.usetex'")
    except Exception as e:
        print(f"Error enabling LaTeX text rendering: {e}")

    try:
        params = {
            'xtick.top': True,
            'ytick.right': True,
            'xtick.direction': 'in',
            'ytick.direction': 'in'
        }
        plt.rcParams.update(params)
        print("Updated tick parameters")
    except Exception as e:
        print(f"Error updating tick parameters: {e}")

# Apply plotting defaults
set_plotting_defaults()

# =============================================
# Helper function: Extract redshift from file path
# =============================================
def extract_redshift_from_file_path(file_path):
    try:
        start_index = file_path.find("z") + 1
        end_index = file_path.find(".txt")
        z_str = file_path[start_index:end_index]
        redshift = float(z_str)
        return redshift
    except ValueError as e:
        print(f"Error extracting redshift from {file_path}: {e}")
        return None

# =============================================
# Main plotting function
# =============================================
def plot_z_powerpectrum(file_paths, zero_file_paths, label_step=5):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = cm.viridis(np.linspace(0, 1, len(file_paths)))  # Generate colors for each plot

    # Extract and store redshifts for sorting
    redshifts = [extract_redshift_from_file_path(file_path) for file_path in file_paths]
    zero_redshifts = [extract_redshift_from_file_path(zero_path) for zero_path in zero_file_paths]

    # Ensure they are sorted
    redshifts_with_paths = sorted(zip(redshifts, file_paths))
    zero_redshifts_with_paths = sorted(zip(zero_redshifts, zero_file_paths))

    # Separate sorted redshifts and paths
    sorted_redshifts, sorted_file_paths = zip(*redshifts_with_paths)
    sorted_zero_redshifts, sorted_zero_file_paths = zip(*zero_redshifts_with_paths)

    for i, (file_path, zero_path, color) in enumerate(zip(sorted_file_paths, sorted_zero_file_paths, colors)):
        z = extract_redshift_from_file_path(file_path)
        if z is None:
            continue

        try:
            data = np.loadtxt(file_path)
            zero_data = np.loadtxt(zero_path)
        except Exception as e:
            print(f"Error loading data from {file_path} or {zero_path}: {e}")
            continue

        frequency = data[:, 0]
        power_model = data[:, 1]
        zero_power = zero_data[:, 1]

        # Calculate the difference with mu0=0.1
        diff_power = power_model / zero_power

        # Plot on log-log scale
        ax.semilogx(frequency, diff_power, color=color, label=f'z={z}')

    ax.set_xlabel(r"$k$ $[h \mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"$B(k)$")
    ax.set_title(r"Power spectrum boost, $\mu_0 =0.1$")
    ax.set_xlim(0.016, 3.14)
    ax.grid(True)

    # Add color bar
    sm = cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=0, vmax=max(sorted_redshifts)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('z')
    
    plt.tight_layout()  # Adjust the spacing between subplots


    # Save the figure
    output_path = "pictures/PSB_z.pdf"
    try:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    except Exception as e:
        print(f"Error saving plot to {output_path}: {e}")

# =============================================
# Main execution
# =============================================
output_folder = '../output/mu0_trial/'
zero_file_paths = glob.glob(os.path.join(output_folder, 'pofk_Sim29_GR_cb*.txt'))
mod_file_paths = glob.glob(os.path.join(output_folder, 'pofk_Sim29_cb*.txt'))

# Ensure the output directory exists before saving
if not os.path.exists("pictures/"):
    os.makedirs("pictures/")

# Run the plotting function
plot_z_powerpectrum(mod_file_paths, zero_file_paths);


# In[19]:


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
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['text.usetex'] = True
params = {'xtick.top': True, 'ytick.right': True, 'xtick.direction': 'in', 'ytick.direction': 'in'}
plt.rcParams.update(params)




# =============================================
# Function to extract the boost
# =============================================
def PofkBoost(file_path, zero_file_paths):
    file_number = int(file_path.split("_")[-3].strip("Sim"))
    zero_file_path = [zp for zp in zero_file_paths if int(zp.split("_")[-4].strip("Sim")) == file_number][0]
    # Load data from files
    data = np.loadtxt(file_path)
    zero_data = np.loadtxt(zero_file_path)

    # Extract frequency and power data
    frequency = data[:, 0]
    power_model = data[:, 1]
    zero_power = zero_data[:, 1]
    
    boost = power_model / zero_power

    return boost, frequency


# =============================================
# Function to print model parameters from lua file
# =============================================
def print_model_parameters(sim_file_path):
    with open(sim_file_path, 'r') as f:
        file_content = f.read()
        gravity_model_marta_mu0 = file_content.split('gravity_model_marta_mu0 = ')[1].split('\n')[0]
        
    model_params = {
        "cosmology_mu0": gravity_model_marta_mu0,
    }
    
    return model_params


# =============================================
# Function to plot the boost
# =============================================
def plot_boost(file_paths, zero_file_paths, lua_dir, parameter):
    fig, ax = plt.subplots(figsize=(10, 6))
    param_values = []
    boost_data = []

    for file_path, zero_path in zip(file_paths, zero_file_paths):
        file_number = int(file_path.split("_")[-3].strip("Sim"))
        lua_path = f'{lua_dir}cola_input_Sim{file_number}.lua'
        parameters = print_model_parameters(lua_path)
        
        boost, frequency = PofkBoost(file_path, zero_file_paths)
        param_key = 'cosmology_' + parameter
        param_values.append(float(parameters[param_key]))
        boost_data.append((frequency, boost))
    
    # Normalize the parameter values to [0, 1] for the color map


    norm = mcolors.Normalize(vmin=min(param_values), vmax=max(param_values))
    cmap = cm.viridis.reversed()    
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Plot each boost curve with color mapped from the parameter value
    for (frequency, boost), param_value in zip(boost_data, param_values):
        color = cmap(norm(param_value))
        ax.semilogx(frequency, boost, color=color)

    # Add a color bar to represent the parameter values
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(r'$\mu_0$')

    ax.set_xlabel(r"$k$ $[h \mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"$B(k)$")
    ax.set_title(r"Power Spectrum boost, $z=0$")
    ax.set_xlim(0.016, 3.14)
    ax.grid(True)

    plt.tight_layout()  # Adjust the spacing between subplots
    plt.savefig(f'pictures/PSB_mu0.pdf')
#    plt.show()


# =============================================
# Main function
# =============================================
def main():
    parameters = ['mu0']

    for parameter in parameters:
        output_folder = f'../output/{parameter}_trial/'
        lua_dir = f'../output_Sim/{parameter}_trial/'
        zero_file_paths = []
        mod_file_paths = []

        for i in range(30):
            zero_file_path = glob.glob(os.path.join(output_folder, f'pofk_Sim{i}_GR_cb_z0.000.txt'))
            mod_file_path = glob.glob(os.path.join(output_folder, f'pofk_Sim{i}_cb_z0.000.txt'))
            mod_file_paths.extend(mod_file_path)  # Use extend instead of append to add file paths to the list
            zero_file_paths.extend(zero_file_path)  # Use extend instead of append to add file paths to the list

        # Remove the GR file from the model file paths
        mod_file_paths = [fp for fp in mod_file_paths if "GR" not in os.path.basename(fp)]
        plot_boost(mod_file_paths, zero_file_paths, lua_dir, parameter)

if __name__ == "__main__":
    main()


# In[19]:



# =============================================
# Function to print model parameters from lua file
# =============================================
def print_model_parameters(sim_file_path):
    try:
        if not os.path.exists(sim_file_path):
            print(f"Lua file not found: {sim_file_path}")
            return {}

        with open(sim_file_path, 'r') as f:
            file_content = f.read()
            parameters = {
                "cosmology_OmegaCDM": float(file_content.split('cosmology_OmegaCDM = ')[1].split('\n')[0]),
                "cosmology_h": float(file_content.split('cosmology_h = ')[1].split('\n')[0]),
                "cosmology_As": float(file_content.split('cosmology_As = ')[1].split('\n')[0]),
                "cosmology_ns": float(file_content.split('cosmology_ns = ')[1].split('\n')[0]),
                "cosmology_Neff": float(file_content.split('cosmology_Neffective = ')[1].split('\n')[0]),
                "cosmology_Omegab": float(file_content.split('cosmology_Omegab = ')[1].split('\n')[0])
            }
        return parameters
    except Exception as e:
        print(f"Error reading {sim_file_path}: {e}")
        return {}

# =============================================
# Function to plot the boost on a given subplot
# =============================================
def plot_boost(ax, file_paths, zero_file_paths, lua_dir, parameter, param_label):
    param_values = []
    boost_data = []

    for file_path in file_paths:
        file_number = int(file_path.split("_")[-3].strip("Sim"))
        lua_path = f'{lua_dir}cola_input_Sim{file_number}.lua'
        
        parameters = print_model_parameters(lua_path)
        if not parameters:
            continue

        boost, frequency = PofkBoost(file_path, zero_file_paths)
        if boost is None or frequency is None:
            continue

        param_key = 'cosmology_' + parameter
        if param_key not in parameters:
            print(f"Parameter key '{param_key}' not found in {lua_path}")
            continue
        
        param_values.append(parameters[param_key])
        boost_data.append((frequency, boost))
    
    if not param_values:
        print(f"No data for parameter '{parameter}'")
        return

    # Normalize the parameter values to [0, 1] for the color map
    norm = mcolors.Normalize(vmin=min(param_values), vmax=max(param_values))
    cmap = cm.viridis
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Plot each boost curve with color mapped from the parameter value
    for (frequency, boost), param_value in zip(boost_data, param_values):
        color = cmap(norm(param_value))
        ax.semilogx(frequency, boost, color=color)

    ax.set_xlabel(r"$k$ $[h \mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"$B(k)$")
    ax.set_title(f"Boost: {param_label}")
    ax.grid(True)

    # Add a color bar to represent the parameter values
    cbar = plt.colorbar(sm, ax=ax)
    #cbar.set_label(r'{}'.format(param_label), labelpad=10)
    #cbar.ax.yaxis.set_label_position('top')
    #cbar.set_label_position('top')  # Move the label to the top of the colorbar


# =============================================
# Main function
# =============================================
def main():
    parameters = [
        ('As', r'$A_s$'),
        ('h', r'$h$'),
        ('Neff', r'$N_\mathrm{eff}$'),
        ('ns', r'$n_s$'),
        ('Omegab', r'$\Omega_b$'),
        ('OmegaCDM', r'$\Omega_\mathrm{CDM}$')
    ]

    n_params = len(parameters)
    nrows = (n_params // 2) + (n_params % 2)
    ncols = 2

    fig, axes = plt.subplots(nrows, ncols, sharex=True, figsize=(15, 5 * nrows))

    for ax, (parameter, param_label) in zip(axes.flatten(), parameters):
        output_folder = f'../output/{parameter}/'
        lua_dir = f'../output_Sim/{parameter}/'
        zero_file_paths = []
        mod_file_paths = []

        for i in range(10):
            zero_file_path = glob.glob(os.path.join(output_folder, f'pofk_Sim{i}_GR_cb_z0.000.txt'))
            mod_file_path = glob.glob(os.path.join(output_folder, f'pofk_Sim{i}_cb_z0.000.txt'))
            mod_file_paths.extend(mod_file_path)  # Use extend instead of append to add file paths to the list
            zero_file_paths.extend(zero_file_path)  # Use extend instead of append to add file paths to the list

        # Remove the GR file from the model file paths
        mod_file_paths = [fp for fp in mod_file_paths if "GR" not in os.path.basename(fp)]

        if not mod_file_paths or not zero_file_paths:
            print(f"No data to plot for parameter '{parameter}'")
            continue

        plot_boost(ax, mod_file_paths, zero_file_paths, lua_dir, parameter, param_label)
        # Only set the x-axis label on the bottom row subplots
        if ax in axes[-1, :]:
            ax.set_xlabel(r"$k$ $[h \mathrm{Mpc}^{-1}]$")
        else:
            ax.set_xlabel("")
        ax.set_xlim(0.016,3.14)

    # Adjust the spacing to make it more compact
    plt.tight_layout()
    # Optionally adjust further if needed
    # fig.subplots_adjust(hspace=0.3)

    plt.savefig('pictures/tuning_parameters_compact.pdf')
if __name__ == "__main__":
    main()


# In[ ]:




