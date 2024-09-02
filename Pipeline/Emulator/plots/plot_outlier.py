import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from figure_size import set_size
import matplotlib.lines as mlines
from Emulator import EmulatorEvaluator
import glob

#=============================================
# Set plotting defaults
#=============================================
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 6})
matplotlib.rcParams['text.usetex'] = True
params = {'xtick.top': True, 'ytick.right': True, 'xtick.direction': 'in', 'ytick.direction': 'in'}
plt.rcParams.update(params)

#=============================================
# A simple class for extracting the boost 
#=============================================
class PofkBoostEmulator:
    def __init__(self, path="", version=0):
        self.evaluator = EmulatorEvaluator.load(path + f"/version_{version}")
    def __call__(self, params):
        inputs = np.array(params)
        return self.evaluator(inputs).reshape(-1)

def get_available_redshifts():
    folder_path    = '/mn/stornext/u3/hansw/Marta/emulator_data_test/'
    file_pattern   = 'pofk_Sim0_cb_z*.txt'
    matching_files = glob.glob(folder_path + '/' + file_pattern)
    numbers_str    = []
    numbers        = []
    for file in matching_files:
        number = file.split('_z')[1].split('.txt')[0]
        numbers.append(float(number))
        numbers_str.append(str(number))
    # sort from highest to lowest
    numbers, numbers_str = zip(*sorted(zip(numbers, numbers_str), reverse=True))
    return numbers_str

def scale_parameter(parameter, min_val, max_val):
    scaled_parameter = (parameter - min_val) / (max_val - min_val) * 2 - 1
    return scaled_parameter

def rescale_parameter(parameter, min_val, max_val):
    scaled_parameter = ((parameter +1)/2)* (max_val - min_val) + min_val
    return scaled_parameter
    

def rescale_parameters_back( params):
    parameters_to_vary = {
        'mu0':          [-0.1, 0.1],
        'Omega_cdm':    [0.2, 0.34],
        'h':            [0.60, 0.64],
        'A_s':          [1.6e-9, 2.6e-9],
        'z':            [0, 20.0],
    }
    params = np.copy(params)  # Make a copy to avoid modifying the original array
    params[0] = rescale_parameter(params[0], parameters_to_vary['mu0'][0], parameters_to_vary['mu0'][1])
    params[1] = rescale_parameter(params[1], parameters_to_vary['Omega_cdm'][0], parameters_to_vary['Omega_cdm'][1])
    params[2] = rescale_parameter(params[2], parameters_to_vary['h'][0], parameters_to_vary['h'][1])
    params[3] = rescale_parameter(params[3], parameters_to_vary['A_s'][0], parameters_to_vary['A_s'][1])
    params[4] = rescale_parameter(params[4], parameters_to_vary['z'][0], parameters_to_vary['z'][1])
    return params


#=============================================
# Set the folder to the emulator and the version
# and set up the boost-function
#=============================================
emulator_folder = './emulators_tuning_8/lightning_logs' 
emulator_version = 1#int(input('version?  '))  
pofkboostfunction = PofkBoostEmulator(path = emulator_folder, version = emulator_version)

#=============================================
# Could get redshift from data, but since we dont 
# have all the data uploaded here, these are the available
# redshifts for the uploaded emulators:
#=============================================

redshifts   = get_available_redshifts()  

z_val = redshifts[-1]

#=============================================
# Load data. The "split" here is to fetch each 
# "P(k)/P(k)_GR" block individually
#=============================================
data_test_  = np.genfromtxt('generate_data/50_separatedLHS_test.csv', delimiter=',')[1:]

split       = 50 #256 is the original # Length of k array basically

data_test0 = []

print (z_val)
for i in range(len(data_test_)//split):
    if (data_test_[i*split:(i+1)*split][0][-3] == scale_parameter(float(z_val[0]),0.000,20.000)):
        data_test0.append(data_test_[i*split:(i+1)*split])

#=============================================
# Set up figures
#=============================================
main_color   = 'grey'
size     = set_size(523.5306, dims=[2, 1], golden_ratio=True,fraction=0.5)
fig, axs = plt.subplots(2,1,figsize=(size[0],size[1]*0.53),\
                        gridspec_kw={'height_ratios':[1.0,0.4]},sharex=True)
mse = []

for i, data_i in enumerate(data_test0): 
    
    log10k = data_i[:,-2]
    k      = 10**log10k
    boost  = data_i[:,-1]

    params_cosmo = np.array([
        data_i[0,0],
        data_i[0,1],    
        data_i[0,2],
        data_i[0,3],
        data_i[0,4],
        ]) 
    params_batch = np.column_stack(( np.vstack([params_cosmo] * len(log10k)), log10k))
    
    boost_emulator = pofkboostfunction(params_batch)
    
    mse.append(np.square(np.subtract(boost, boost_emulator)).mean())
idx_max = mse.index(max(mse))
for i, data_i in enumerate(data_test0): 
    
    #=============================================
    # Extract data from test file, first redshift
    #=============================================
    log10k = data_i[:,-2]
    k      = 10**log10k
    boost  = data_i[:,-1]

    #=============================================
    # Fetch parameters from the trainingset
    #=============================================
    params_cosmo = np.array([
        data_i[0,0],
        data_i[0,1],    
        data_i[0,2],
        data_i[0,3],
        data_i[0,4],
        ]) 
    params_batch = np.column_stack(( np.vstack([params_cosmo] * len(log10k)), log10k))
 
    #=============================================
    # Call emulator
    #=============================================
    boost_emulator = pofkboostfunction(params_batch)
    
    #=============================================
    # Make plot
    #=============================================
    if i != idx_max:
        axs[0].semilogx(k, boost_emulator, color=main_color, linewidth=0.6,linestyle = '--', alpha=0.5)
        axs[1].semilogx(k, boost_emulator/boost - 1.0, color=main_color, linewidth=0.6, alpha=0.5)
        axs[0].semilogx(k,boost,color=main_color,alpha=0.5,linewidth=0.6)
    else:
        axs[0].semilogx(k, boost_emulator, color='green', linewidth=0.6,linestyle = '--', alpha=0.5)
        axs[1].semilogx(k, boost_emulator/boost - 1.0, color='green', linewidth=0.6, alpha=0.5)
        axs[0].semilogx(k,boost,color='green', linewidth = 0.6, alpha=0.5)
        rescaled_params = rescale_parameters_back(params_cosmo)
        print(rescaled_params)

#=============================================
# Prettify the plots
#=============================================
axs[1].set_ylabel(r'$\displaystyle \mathrm{Rel. Diff.}$', fontsize=6)
axs[0].set_ylabel(r"$B(k,z) = P_{f(R)}/P_{\rm GR}$", fontsize=6)

axs[1].set_xlabel(r'$k\,\;[h\,\mathrm{Mpc}^{-1}]$')


axs[1].fill_between(k, k*0-0.0025,k*0+0.0025,color='gray',alpha=0.15,linewidth=0.0)
axs[1].set_ylim(-0.008,0.008)

axs[0].set_xlim(min(k),max(k))


data_patch = mlines.Line2D([0,0],[0,0],color=main_color, alpha=0.5,label='data')
emu_patch  = mlines.Line2D([0,0],[0,0],color=main_color,linestyle='--',alpha=0.5, label='emulator')
axs[0].legend(handles=[data_patch,emu_patch],frameon=False,fontsize=6,labelspacing=0.2,loc='upper left')


axs[0].text(0.03,1-0.04,'z = '+z_val[0])

plt.subplots_adjust(wspace=0, hspace=0)
fig.align_ylabels()

#=============================================
# Show plot
#=============================================
plt.show()

#=============================================
# Save the plot as PDFs
#=============================================
figpath = f'plots/cm_ver{emulator_version}_Pk_outlier.pdf'
print("Saving plot to ", figpath)
fig.savefig(figpath, format='pdf', bbox_inches='tight')
