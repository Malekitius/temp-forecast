# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from dataset import unscale_temperature

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_true_pred_recurrent(model, test_dataset):
    """
    Returns unscaled true and predicted values for all blocks in full_dataset

    Parameters
    ----------
    model : model instance
        Model to return predicted values
    test_dataset : Heat_Sequence_Dataset instance
        Dataset to be loaded into the model for gathering predictions

    Returns
    ----------
    y_true : list of torch.tensor
        True dataset values
    y_pred : list of torch.tensor
        Predicted dataset values
    """
    seq, y = test_dataset[0]
    seq, y = seq.to(device), y.to(device)

    y_true = []
    y_pred = []

    y_true += [unscale_temperature(y)]
    prediction = model(seq.unsqueeze(dim=0)).squeeze(dim=0)
    y_pred += [unscale_temperature(prediction)]

    with torch.no_grad():
        for i in range(1, len(test_dataset)):
            x, y = test_dataset[i]
            x, y = x.to(device), y.to(device)

            y_true += [unscale_temperature(y)]
            x = x[-1,-2:]

            new_row = torch.cat((prediction, x)).unsqueeze(dim=0)
            seq = torch.cat((seq[1:, :], new_row), axis=0)

            prediction = model(seq.unsqueeze(dim=0)).squeeze(dim=0)

            y_pred += [unscale_temperature(prediction)]

    return y_true, y_pred



def get_block_temperature_recurrent(model, test_dataset, block_number):
    """
    Returns true and predicted values for one block in test_dataset

    Parameters
    ----------
    model : model instance
        Model to return predicted values
    test_dataset : Heat_Sequence_Dataset instance
        Dataset to be loaded into the model for gathering predictions
    block_number : int
        Number of block to gather true and predicted values for

    Returns
    ----------
    y_block_true : list of float
        True dataset values
    y_block_pred : list of float
        Predicted dataset values
    """
    y_block_true = []
    y_block_pred = []

    y_true, y_pred = get_true_pred_recurrent(model, test_dataset)

    for i in range(len(y_true)):
        y_block_true += [y_true[i][block_number - 1].item()]
        y_block_pred += [y_pred[i][block_number - 1].item()]
    return y_block_true, y_block_pred



def plot_prediction_recurrent(model, test_dataset, block_number, mode='Validation'):
    """
    Plots true and one-step-predicted time series for the specified block

    Parameters
    ----------
    model : model instance
        Model to return predicted values
    test_dataset : Heat_Sequence_Dataset instance
        Dataset to be loaded into the model for gathering predictions
    block_number : int
        Number of block to gather true and predicted values for
    mode : str
        String to specify whether this function is used for validation or test dataset
    """
    y_block_true, y_block_pred = get_block_temperature_recurrent(model, test_dataset,
                                                                 block_number)

    plt.figure(figsize=(5, 5), dpi=150)

    plt.plot(np.arange(len(y_block_pred)), y_block_true, label='True', color='black')

    plt.plot(np.arange(len(y_block_pred)), y_block_pred, label='Pred', color='red')

    plt.ylabel(f'Temperature of block N{block_number}')
    plt.xlabel('Time ticks (10 s)')
    plt.title(f'{mode} recurrent prediction: temperature of block N{block_number} from time')
    plt.legend()
    plt.grid()
    plt.show()