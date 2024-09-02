import optuna
import os
import glob
import yaml
import argparse
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as TorchFunctional
from pydantic import BaseModel
from typing import Tuple, List, Union
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataset import Dataset
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt       
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
import torchvision


SEED = 13
torch.manual_seed(SEED)
tuning_study = False # Looking for the best hyperpameters?


class Parameters(BaseModel):
    feature_columns:             List[str]
    label_columns:               List[str]
    hidden_layers:               List[int]
    trainingdata_csvfile_path:   "str"
    validationdata_csvfile_path: "str"
    testdata_csvfile_path:       "str"
    max_epochs:                  int
    batch_size:                  int
    nthreads:                    int
    activation_function:         str
    loss_function:               str
    output_dir:                  str
    learning_rate:               float
    weight_decay:                float

class Net(pl.LightningModule):
    def __init__(self, n_features, output_dim, learning_rate, weight_decay, hidden_layers, loss_function, activation_function, **kwargs):
        super(Net, self).__init__()
        self.learning_rate       = learning_rate
        self.weight_decay        = weight_decay
        self.n_hidden_layers     = len(hidden_layers)
        self.n_features          = n_features
        self.output_dim          = output_dim
        self.loss_function       = getattr(torch.nn, loss_function)()
        self.activation_function = getattr(TorchFunctional, activation_function)
        self.hidden_layers       = nn.ModuleList()
        self.activations         = nn.ModuleList()
        self.output_layer        = nn.Linear(in_features=hidden_layers[-1], out_features=self.output_dim)
        self.dropout             = nn.Dropout(p=0.0)
        for i in range(self.n_hidden_layers):
            layer = nn.Linear(in_features=self.n_features if i == 0 else hidden_layers[i - 1], out_features=hidden_layers[i])
            self.hidden_layers.append(layer)
        self.save_hyperparameters("n_features","output_dim","learning_rate","weight_decay","hidden_layers","loss_function","activation_function")
    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers):
            x = self.activations[i](layer.forward(x)) if self.activations else self.activation_function(layer.forward(x))
        output = self.output_layer.forward(x)
        return output
    def custom_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x,)
        return self.loss_function(y_hat, y)
    def training_step(self, batch, batch_idx):
        loss = self.custom_step(batch=batch, batch_idx=batch_idx)
        self.log("train_loss", loss)
        return loss
    def test_step(self, batch, batch_idx):
        loss = self.custom_step(batch=batch, batch_idx=batch_idx)
        self.log("test_loss", loss)
        return loss
    def validation_step(self, batch, batch_idx):
        loss = self.custom_step(batch=batch, batch_idx=batch_idx)
        self.log("val_loss", loss)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=40, factor=0.1, min_lr=1.0e-6, verbose=True)
        return { "optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch", "frequency": 1} }



class PyTorchDataset(torch.utils.data.Dataset):
    def __init__(self, features: Union[np.array, torch.tensor], labels: Union[np.array, torch.tensor]):
        self.features, self.labels = features, labels
    def __len__(self,) -> int:
        return len(self.features)
    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:
        return (self.features[index, :], self.labels[index, :])

class PyTorchDataModule(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.feature_columns = params.feature_columns
        self.label_columns   = params.label_columns
        self.batch_size      = params.batch_size
        self.nthreads        = params.nthreads
        self.training_data   = self.load_csv_file(params.trainingdata_csvfile_path)
        self.test_data       = self.load_csv_file(params.testdata_csvfile_path)
        self.validation_data = self.load_csv_file(params.validationdata_csvfile_path)
    def load_csv_file(self, path) -> "Dataset":
        df = pd.read_csv(path)
        return PyTorchDataset(features=torch.from_numpy(df[self.feature_columns].to_numpy().astype(np.float32)), labels=torch.from_numpy(df[self.label_columns].to_numpy().astype(np.float32)))
    def train_dataloader(self,) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.training_data, batch_size=self.batch_size, num_workers=self.nthreads, shuffle=True)
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.validation_data, num_workers=self.nthreads, batch_size=self.batch_size)
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.nthreads)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0., std=1.e-3)
        m.bias.data.fill_(0)

class EmulatorEvaluator:
    def __init__(self, model):
        self.model = model
    @classmethod
    def load(self, path):
        checkpointpath = [file for file in glob.glob(path + "/checkpoints/*.ckpt")][0]
        with open(path + "/hparams.yaml", "r") as fp:
            hyperparams = yaml.safe_load(fp)
        model = Net.load_from_checkpoint(checkpointpath, **hyperparams)
        return self(model=model)
    def __call__(self, inputs: Union[np.array, torch.tensor]): 

        inputs = torch.from_numpy(inputs.astype(np.float32))
        with torch.no_grad():
            predictions = self.model(inputs)
        return predictions.detach().numpy()


def objective(trial):
    # Modify hyperparameters based on trial's suggestions
    param.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2,log=True)
    param.weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-5,log=True)
    param.batch_size = trial.suggest_categorical('batch_size', [8,16,32])
    
 
    # Initialize datamodule and model
    datamodule = PyTorchDataModule(param)
    model = Net(n_features=datamodule.training_data.features.shape[-1],
                output_dim=datamodule.training_data.labels.shape[-1],
                **dict(param))
    
    # Initialize trainer with callbacks and hyperparameters
    callbacks = [EarlyStopping(monitor="val_loss", patience=40, mode="min", verbose=False),StochasticWeightAveraging(1e-2)]
    trainer = pl.Trainer(callbacks=callbacks, 
                         max_epochs=param.max_epochs, 
                         default_root_dir=param.output_dir,
                         gradient_clip_val=0.5)

    # Fit and test the model
    trainer.fit(model=model, datamodule=datamodule)
    result = trainer.validate(model=model, datamodule=datamodule)
    
    # We return the validation loss as the objective value to minimize
    val_loss = result[0]['val_loss']
    return val_loss


# Function to rescale a parameter to the range [-1, 1]
def rescale_parameter(parameter, min_val, max_val):
    parameter = np.array(parameter)
    scaled_parameter = ((parameter - min_val) / (max_val - min_val)) * 2 - 1
    return scaled_parameter

class PofkBoostEmulator:
    def __init__(self, path="", version=0):
        self.evaluator = EmulatorEvaluator.load(path + f"/version_{version}")
        self.parameters_to_vary = {
            'mu0':        [-0.1, 0.1],
            'Omega_cdm':  [0.2, 0.34],
            'h':          [0.60, 0.74],
            'A_s':        [1.6e-9, 2.6e-9],
            'z':          [0, 20.0],
        }
    def _rescale_parameters(self, params):
        params = np.copy(params)  # Make a copy to avoid modifying the original array
        params[0] = rescale_parameter(params[0], *self.parameters_to_vary['mu0'])
        params[1] = rescale_parameter(params[1], *self.parameters_to_vary['Omega_cdm'])
        params[2] = rescale_parameter(params[2], *self.parameters_to_vary['h'])
        params[3] = rescale_parameter(params[3], *self.parameters_to_vary['A_s'])
        params[4] = rescale_parameter(params[4], *self.parameters_to_vary['z'])
        return params

    def __call__(self, params):
        # Assumes `params` is an array with shape (n_samples, n_features)
        scaled_params = np.apply_along_axis(self._rescale_parameters, 1, params)
        return self.evaluator(scaled_params).reshape(-1)

if __name__ == "__main__":
    with open("input.yaml", "r") as f:
        inputfile = yaml.safe_load(f)
    param = Parameters(**inputfile)
    if (tuning_study):
        # Optuna study for hyperparameter optimization
        study = optuna.create_study(
                    study_name="hyperparameter_study_test_0", 
                    storage="sqlite:///example.db",
                    direction='minimize')
        study.optimize(objective,  n_trials=20,timeout=6000)  
        # After the study, you can get and save the best trial's hyperparameters
        best_trial = study.best_trial
        print("Best trial:")
        for key, value in best_trial.params.items():
            print(f"{key}: {value}")
        best_params = best_trial.params
        param.learning_rate = best_params['learning_rate']
        param.weight_decay = best_params['weight_decay']
        #param.hidden_layers = best_params['hidden_layers']
        param.batch_size = best_params['batch_size']
    # Initialize datamodule and model with the best hyperparameters
    datamodule = PyTorchDataModule(param)
    model = Net(n_features=datamodule.training_data.features.shape[-1],
                output_dim=datamodule.training_data.labels.shape[-1],
                **dict(param))

    callbacks = [EarlyStopping(monitor="val_loss", patience=30, mode="min", verbose=False),
                 StochasticWeightAveraging(1e-2)]
    trainer = pl.Trainer(callbacks=callbacks,
                         max_epochs=param.max_epochs,
                         default_root_dir=param.output_dir,
                         gradient_clip_val=0.5)
    trainer.fit(model=model, datamodule=datamodule)
    result = trainer.test(model=model, datamodule=datamodule)

    # We return the validation loss as the objective value to minimize
    test_loss = result[0]['test_loss']
    #print('val_loss: ',val_loss)
    if (tuning_study):
        # Optional: Fit the model with the best hyperparameters
        os.makedirs('best_input', exist_ok=True)
        with open("best_input/input_test_best.yaml", "w") as f:
            yaml.dump(dict(best_trial.params), f)

        best_test_result = trainer.test(model=model, datamodule=datamodule)
        # Plot and save the optimization history
        fig1 = plot_optimization_history(study)
        fig1.write_image('optimization_history.png')
	    # Plot and save the parameter importances as an image
        fig2 = plot_param_importances(study)
        fig2.write_image('param_importance.png')
        # If you want to display the figures as well, after saving.
        plt.show()  # Uncomment this line if needed.
        print("Test results with best hyperparameters:")
        print(best_test_result)

