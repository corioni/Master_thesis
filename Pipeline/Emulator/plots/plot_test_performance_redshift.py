import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from figure_size import set_size
import matplotlib.lines as mlines
from Emulator import EmulatorEvaluator
import glob

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

def rescale_parameter(parameter, min_val, max_val):
    scaled_parameter = (parameter - min_val) / (max_val - min_val) * 2 - 1
    return scaled_parameter

#=============================================
# Set plotting defaults
#=============================================
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 10})
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

z_val = [redshifts[-25],redshifts[-12],redshifts[-2]]

#=============================================
# Load data. The "split" here is to fetch each 
# "P(k)/P(k)_GR" block individually
#=============================================
data_test_  = np.genfromtxt('generate_data/50_separatedLHS_test.csv', delimiter=',')[1:]

split       = 50 #256 is the original # Length of k array basically

data_test0 = []; data_test1  = [];data_test2  = []

print (z_val)
for i in range(len(data_test_)//split):
    if (data_test_[i*split:(i+1)*split][0][-3] == rescale_parameter(float(z_val[0]),0.000,20.000)):
        data_test0.append(data_test_[i*split:(i+1)*split])
    if (data_test_[i*split:(i+1)*split][0][-3] == rescale_parameter(float(z_val[1]),0.000,20.000)):
        data_test1.append(data_test_[i*split:(i+1)*split])
    if (data_test_[i*split:(i+1)*split][0][-3] == rescale_parameter(float(z_val[2]),0.000,20.000)):
        data_test2.append(data_test_[i*split:(i+1)*split])

#=============================================
# Set up figures
#=============================================
colors   = ['#DDCC77','#882255','#6699CC']
size     = set_size(523.5307, dims=[8, 1], golden_ratio=True,fraction=0.5)
fig, axs = plt.subplots(8,1,figsize=(size[0],size[1]*0.53),\
                        gridspec_kw={'height_ratios':[1.0,0.4,0.05,1.0,0.4,0.05,1.0,0.4]},sharex=True)

for i, data_i in enumerate(data_test0): 
    
    #=============================================
    # Extract data from test file, first redshift
    #=============================================
    log10k = data_i[:,-2]
    k      = 10**log10k
    boost  = data_i[:,-1]
    axs[0].semilogx(k,boost,color=colors[2],alpha=0.5,)

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
    axs[0].semilogx(k, boost_emulator, color=colors[1], linewidth=0.7, alpha=0.5)
    axs[1].semilogx(k, boost_emulator/boost - 1.0, color=colors[0], linewidth=0.7, alpha=0.5)

for i, data_i in enumerate(data_test1): 
    
    #================================================
    # Extract data from test file, second redshift
    #================================================
    log10k = data_i[:,-2]
    k      = 10**log10k
    boost  = data_i[:,-1]
    axs[3].semilogx(k,boost,color=colors[2],alpha=0.5,)
    
    #================================================
    # Fetch parameters from the validation set
    #================================================
    params_cosmo = np.array([
          data_i[0,0],
          data_i[0,1],    
          data_i[0,2],
          data_i[0,3],
          data_i[0,4],
          ])
    params_batch = np.column_stack(( np.vstack([params_cosmo] * len(log10k)), log10k))
    
    #================================================
    # Call emulator
    #================================================
    boost_emulator = pofkboostfunction(params_batch)
    
    #=============================================
    # Make plot
    #=============================================
    axs[3].semilogx(k, boost_emulator, color=colors[1], linewidth=0.7, alpha=0.5)
    axs[4].semilogx(k, boost_emulator/boost - 1.0, color=colors[0], linewidth=0.7, alpha=0.5)

for i, data_i in enumerate(data_test2): 
    
    #=============================================
    # Extract data from test file, third redshift
    #=============================================
    log10k = data_i[:,-2]
    k      = 10**log10k
    boost  = data_i[:,-1]
    axs[6].semilogx(k,boost,color=colors[2],alpha=0.5,)
    
    #=============================================
    # Fetch parameters from the test set
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
    axs[6].semilogx(k, boost_emulator, color=colors[1], linewidth=0.7, alpha=0.5)
    axs[7].semilogx(k, boost_emulator/boost - 1.0, color=colors[0], linewidth=0.7, alpha=0.5)
#=============================================
# Prettify the plots
#=============================================
axs[1].set_ylabel(r'$\displaystyle \mathrm{Rel. Diff.}$', fontsize=7)
axs[0].set_ylabel(r"$B(k,z) = P_{f(R)}/P_{\rm GR}$", fontsize=7)
axs[4].set_ylabel(r'$\displaystyle \mathrm{Rel. Diff.}$', fontsize=7)
axs[3].set_ylabel(r"$B(k,z) = P_{f(R)}/P_{\rm GR}$", fontsize=7)
axs[7].set_ylabel(r'$\displaystyle \mathrm{Rel. Diff.}$', fontsize=7)
axs[6].set_ylabel(r"$B(k,z) = P_{f(R)}/P_{\rm GR}$", fontsize=7)


axs[7].set_xlabel(r'$k\,\;[h\,\mathrm{Mpc}^{-1}]$')


axs[1].fill_between(k, k*0-0.0025,k*0+0.0025,color='gray',alpha=0.15,linewidth=0.0)
axs[4].fill_between(k, k*0-0.0025,k*0+0.0025,color='gray',alpha=0.15,linewidth=0.0)
axs[7].fill_between(k, k*0-0.0025,k*0+0.0025,color='gray',alpha=0.15,linewidth=0.0)
axs[1].set_ylim(-0.008,0.008)
axs[4].set_ylim(-0.008,0.008)
axs[7].set_ylim(-0.008,0.008)

#axs[0].set_xlim(min(k),max(k))
axs[2].set_visible(False)
axs[5].set_visible(False)


data_patch = mlines.Line2D([0,0],[0,0],color=colors[2], alpha=0.5,label='data')
emu_patch  = mlines.Line2D([0,0],[0,0],color=colors[1],alpha=0.5, label='emulator')
axs[0].legend(handles=[data_patch,emu_patch],frameon=False,fontsize=7,labelspacing=0.2,loc='upper left')

axs[0].set_xlim(min(k),max(k))

axs[0].text(0.03,1-0.04,'z = '+z_val[0])
axs[3].text(0.03,1-0.02,'z = '+z_val[1])
axs[6].text(0.03,1-0.07,'z = '+z_val[2])

plt.subplots_adjust(wspace=0, hspace=0)
fig.align_ylabels()

#=============================================
# Show plot
#=============================================
plt.show()

#=============================================
# Save the plot as PDFs
#=============================================
figpath = f'plots/cm_ver{emulator_version}_Pk_3zs.pdf'
print("Saving plot to ", figpath)
fig.savefig(figpath, format='pdf', bbox_inches='tight')
