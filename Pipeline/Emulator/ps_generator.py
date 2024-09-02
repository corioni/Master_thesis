import euclidemu2 
import numpy as np
import matplotlib.pyplot as plt
import scipy, scipy.interpolate
from Emulator import EmulatorEvaluator
from Emulator import PofkBoostEmulator

import numpy as np

class PowerSpectrumBooster:

    def __init__(self, emulator_path, emulator_version):
        self.pofkboost = PofkBoostEmulator(path=emulator_path, version=emulator_version)

    def compute_power_spectrum(self, cosmological_parameters, redshifts, k_values):
        ee2 = euclidemu2.PyEuclidEmulator()
        k, pnl, plin, b = ee2.get_pnonlin(cosmological_parameters, redshifts, k_values)
        return k, pnl, plin

    def compute_boost_factor(self, cosmological_parameters,mu0, redshifts, k_values):
        log10k = np.log10(k_values)
        Omega_cdm = cosmological_parameters['Omm']- cosmological_parameters['Omb'] - (cosmological_parameters['mnu'] /( 93.14*( cosmological_parameters['h'] **2 )))       
        params_cosmo = np.array([
            mu0,
            Omega_cdm,#cosmological_parameters['Omm'],
            cosmological_parameters['h'],
            cosmological_parameters['As'],
            redshifts[0]
        ])
        params_batch = np.column_stack((np.vstack([params_cosmo] * len(log10k)), log10k)) 
        return self.pofkboost(params_batch)

    def boosted_power_spectrum(self, cosmological_parameters,mu0, redshifts, k_values):
        k, pnl, plin = self.compute_power_spectrum(cosmological_parameters, redshifts, k_values)
        boost_emulator = self.compute_boost_factor(cosmological_parameters,mu0, redshifts, k)
        boosted_plin = boost_emulator * plin[0]
        boosted_pnl = boost_emulator * pnl[0]
        return k, boosted_plin, boosted_pnl, boost_emulator


def main():
    emulator_folder = './emulators_tuning_8/lightning_logs'
    pofkboostfunction_version = 1
    cosmo_par = {'As': 2.1e-09, 'ns': 0.966, 'Omb': 0.04, 'Omm': 0.3, 'h': 0.68, 'mnu': 0.15, 'w': -1.0, 'wa': 0.0}
    redshifts = [0.3, 2, 4, 6, 8, 10]
    k_custom = np.geomspace(0.04, np.pi, 100)

    booster = PowerSpectrumBooster(emulator_folder, pofkboostfunction_version)
    k, boosted_plin, boosted_pnl, boost_emulator = booster.boosted_power_spectrum(cosmo_par, redshifts, k_custom)

    # Save data for z=0 to a text file
    data = np.column_stack((k, boosted_plin, boosted_pnl, boost_emulator))
    np.savetxt("boosted_power_spectra_z0.txt", data, header="k Boosted_P_lin(z=0) Boosted_P_nl(z=0) Boost_factor", fmt='%e')

if __name__ == "__main__":
    main()
