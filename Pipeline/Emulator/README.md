
## Overview

This directory houses all the essential components for employing the emulator and plotting its results. 

As we try to understand the universe and its expansion without relying on the concept of dark energy, which is a big part of the widely accepted cosmic model known as ΛCDM, they are exploring new theories of gravity. These newer ideas, which are different from the ΛCDM model, change the way galaxies and other structures grow over time. We can see these changes in something called the matter power spectrum of perturbation.

The power spectrum of perturbations refers to the distribution of the amplitude of fluctuations in the density of matter or radiation across different scales. These fluctuations are believed to have originated from quantum mechanical fluctuations in the early universe, which later grew under the influence of gravity.
Understanding these differences requires complex computer simulations of the universe, but these simulations are very resource-intensive and are tailored to a specific set of cosmic rules. If those rules change even slightly, the whole simulation needs to be run from the start.

To avoid running numerous simulations, we can create special tools called emulators. These emulators can quickly predict what the matter power spectrum might look like under different conditions. By analyzing this data with the help of emulators, we hope to test and refine alternative theories to the ΛCDM model.

In the context of investigating alternative gravitational theories, we introduce an emulator designed to compute deviations in the cosmic matter Power spectrum. This computational model is specialized to forecast how alterations to gravity, parameterized by specific modifications, affect the distribution of matter compared to predictions based on General Relativity (GR). The emulator calculates the ratio of the Power spectrum under a modified theory of gravity to that derived from GR, providing a quantitative measure of the gravitational modifications' impact.The sophistication of the emulator lies in its ability to do this across an array of some key cosmological parameters, effectively mapping out a multidimensional landscape of potential outcomes. Such capability significantly enhances our capacity to systematically scrutinize different models of modified gravity and their influence on cosmic structure, while simultaneously circumventing the limitations associated with direct, high-resolution cosmological simulations for each parameter set. This tool thereby serves as an essential asset for the in-depth analysis of weak-lensing survey data and for constraining beyond-ΛCDM theories of the cosmos.

The core script `Emulator.py` facilitates the construction and training of our neural network-based model, with an added functionality for hyperparameter optimization. The configurations producing different variations of the trained models are retained within `emulators_tuning*` directories.

## Plotting Scripts

### `plot_train_test_val_performance.py`
This script plots the emulator's predictions against the training, testing, and validation datasets within the context of the unscreened linear case at redshift `z=0.0`.

### `plot_test_performance_redshifts.py`
This script illustrates the emulator's prediction accuracy by comparing its outputs with test dataset samples at three distinct redshift values, serving to highlight the emulator's efficacy across different cosmological conditions.

## Getting Started

### Requirements
Before running the emulator and plotting scripts, ensure that the following software and libraries are installed:
- Python 
Python modules needed to use the emulator:
- yaml
- torch
- numpy
- pandas
- pytorch_lightning
- pydantic
- typing
- scikit-learn

### Data
The data we use to train end test the emulator is aquired though N-Body simulations after generating the input in a hypercube in the parameters space. ...

### Usage
To utilize the emulator and generate plots, follow these instructions:
1. Set the hyperparameters in the input.yaml file
2. Run the `Emulator.py` script to train the model. This script can be run with the following command:
    ```
    python Emulator.py
    ```
3. To visualize the emulator performance, modify the version of the emulator you want to use and than useexecute the plotting scripts:
    ```
    python plot_train_test_val_performance.py
    python plot_test_performance_redshifts.py
    ```
   
The resulting plots will be saved in `plots`.


