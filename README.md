# PNL Prediction for parametrized gravity Pipeline

This repository provides a fully functional pipeline for predicting the non-linear matter-power spectrum within parametrized gravity model. The pipeline leverages the COLA (COmoving Lagrangian Acceleration) method for simulations and applies machine learning tools to train an emulator using simulation data.

## Features

- Simulation code using the COLASolver from the **FML library**.
- Neural network training performed with **PyTorch Lightning**, a lightweight wrapper for the PyTorch module.

## Repository Structure

The repository is organized into the following folders:

### 1. `FML/`
Contains all scripts and configurations required to run the COLA simulations using the COLASolver implemented for including the parametrized gravity model.

### 2. `Pipeline/`
Includes the machine learning component of the pipeline with scripts for training the neural network using PyTorch Lightning. This folder contains all the necessary code to:

- Draw parameter samples using Latin hypercube sampling.
- Create runfiles for the COLASolver code for all the samples.
- Convert COLASolver output data into a format suitable for neural network training.
- Prepare the input files for neural network training.
- Execute the neural network training process.




