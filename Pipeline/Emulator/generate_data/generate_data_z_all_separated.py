import numpy as np
import json
import glob
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import random
from scipy.signal import savgol_filter
import random

def get_available_redshifts():
    folder_path    = '/mn/stornext/u3/hansw/Marta/emulator_data_test/'#'/uio/hume/student-u24/martacor/Simulation/FML/FML/COLASolver/output/emulator_data/'
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

def spline_ratio(k,ratio,n):
    cs        = CubicSpline(k, ratio)
    new_k     = np.logspace(np.log10(min(k)),np.log10(max(k)),n)
    new_ratio = cs(new_k)
    return new_k, new_ratio

def read_file(N_sample,z,set_name,set_prefix):
    home      = '/mn/stornext/u3/hansw/Marta/emulator_data_test/'#'/uio/hume/student-u24/martacor/Simulation/FML/FML/COLASolver/output/emulator_data/'
    ratio     = []
    ratio_lin = []
    #read data:
    k,Pk,Pk_lin ,P_nl         = np.loadtxt(home+'pofk_Sim'+str(N_sample)+'_cb_z'+z+'.txt',unpack=True)
    k_GR,Pk_GR,Pk_lin_GR,P_nl_GR = np.loadtxt(home+'pofk_Sim'+str(N_sample)+'_GR_cb_z'+z+'.txt',unpack=True)
    #calculate ratios:
    r     = Pk/Pk_GR
    r_lin = Pk_lin/Pk_lin_GR
    #spline to get log-spaced k-array:
    new_k, new_r         = spline_ratio(k,r,50)
    new_k_lin, new_r_lin = spline_ratio(k,r_lin,50)
    #smooth data:
    new_r     = savgol_filter(new_r, 50, 3)
    new_r_lin = savgol_filter(new_r_lin, 50, 3)
    #append all samples to array:
    ratio.append(new_k)
    ratio.append(new_r)
    ratio_lin.append(new_k_lin)
    ratio_lin.append(new_r_lin)
    return ratio

# Function to rescale a parameter to the range [-1, 1]
def rescale_parameter(parameter, min_val, max_val):
    parameter = np.array(parameter)
    scaled_parameter = ((parameter - min_val) / (max_val - min_val)) * 2 - 1
    return scaled_parameter


def read_json(filename):
    # Read the JSON data from a file
    with open(filename) as file:
        json_data = json.load(file)

    mu0 = []; Omegacdm = []; h = []; As = [];

    for entry in json_data.values():
        mu0.append(entry['cosmo_param']['gravity_model_marta_mu0'])
        Omegacdm.append(entry['cosmo_param']['Omega_cdm'])
        h.append(entry['cosmo_param']['h'])
        As.append(entry['cosmo_param']['A_s'])
    return mu0,Omegacdm,h,As

redshifts   = get_available_redshifts()
sets_name   = ['training','testing','validation']
sets_prefix = ['train','test','val']
sets_num    = [0,256,288,320]#[80,90,100] 
parameters_to_vary = {
    'gravity_model_marta_mu0':  [-0.1,0.1],
    'Omega_cdm':    [0.2,0.34],  # +-25%, euclid has 20%
    'h':            [0.60,0.74],  # +-10% Euclid has 9%
    'A_s':         [1.6e-9,2.6e-9], # +-25%, Euclid has +-20%
    'z':            [0,20.0],
}

for i in range(len(sets_name)):

    mu0_list = []; Omegacdm_list  = []; h_list = []
    As_list = [];  z_list  = []
    k_list  = []; ratio_list     = []

    jsonfile  = '/uio/hume/student-u24/martacor/Simulation/Pipeline/parameters_Sim_320.json'
    mu0, Omegacdm,  h, As = read_json(jsonfile)
    headers = ["mu0", "Omegacdm","h","As","z", "log10k", "boost"] #  "h", "As",'z'
    

    for z in range(len(redshifts)):
        for s in range(sets_num[i],sets_num[i+1]):
        #while s<sets_num[i]:
            print ('Appending redshift ', redshifts[z], ' for sample ', s, 'for data set ', sets_name[i])
            ratio = read_file(s,redshifts[z],sets_name[i],sets_prefix[i])
            mu0_list.extend(np.zeros(len(ratio[0]))+mu0[s])
            Omegacdm_list.extend(np.zeros(len(ratio[0]))+Omegacdm[s])
            h_list.extend(np.zeros(len(ratio[0]))+h[s])
            As_list.extend(np.zeros(len(ratio[0]))+As[s])
            z_list.extend(np.zeros(len(ratio[0]))+float(redshifts[z]))
            k_list.extend(np.log10(ratio[0]))
            ratio_list.extend(ratio[1])

    mu0_list = rescale_parameter(mu0_list, parameters_to_vary['gravity_model_marta_mu0'][0], parameters_to_vary['gravity_model_marta_mu0'][1])
    Omegacdm_list = rescale_parameter(Omegacdm_list, parameters_to_vary['Omega_cdm'][0], parameters_to_vary['Omega_cdm'][1])
    h_list = rescale_parameter(h_list, parameters_to_vary['h'][0], parameters_to_vary['h'][1])
    As_list = rescale_parameter(As_list, parameters_to_vary['A_s'][0], parameters_to_vary['A_s'][1])
    z_list = rescale_parameter(z_list, parameters_to_vary['z'][0], parameters_to_vary['z'][1])
    data = np.zeros((len(mu0_list),len(headers)))
    
    data[:,0] = mu0_list
    data[:,1] = Omegacdm_list
    data[:,2] = h_list
    data[:,3] = As_list
    data[:,-3] = z_list
    data[:,-2] = k_list
    data[:,-1] = ratio_list

    np.savetxt('50_separatedLHS_'+sets_prefix[i]+'.csv', data, delimiter=",", header=",".join(headers), comments="")
