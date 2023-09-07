# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output


def plot_dataset(heat_data, block_number):
    """
    Plots the whole time series dataset

    Parameters
    ----------
    heat_data : pd.DataFrame
        Dataframe containing time series dataset
    block_number: int
        Number of block to plot temperature time series for
    """
    plt.figure(dpi=120)

    plt.plot(np.arange(len(heat_data)), heat_data.loc[:,block_number])

    plt.grid()
    plt.xlabel('time')
    plt.ylabel('temperature')
    plt.title(f'block number = {block_number}')
    plt.show()



def plot_smoothed_data(heat_data, heat_data_smoothed, block_number):
    """
    Plots both non-filtered and filtered time series in parallel on one plot

    Parameters
    ----------
    heat_data : pd.DataFrame
        Dataframe containing non-filtered dataset
    heat_data_smoothed : pd.DataFrame
        Dataframe containing filtered dataset
    block_number: int
        Number of block to plot temperature time series for
    """
    clear_output()
    plt.figure(figsize=(5, 5), dpi=120)

    plt.plot(range(1, len(heat_data) + 1), heat_data.iloc[:, block_number], label='original', color='black')

    plt.plot(range(1, len(heat_data) + 1), heat_data_smoothed[:, block_number], label='smoothed', color='red')

    plt.ylabel('Temperature')
    plt.xlabel('Time')
    plt.title(f'block_number = {block_number}')
    plt.legend()
    plt.grid()
    plt.show()



def plot_train_val_data(heat_data, train_size, block_number):
    """
    Plots train and validation time series consequently on one plot

    Parameters
    ----------
    heat_data : pd.DataFrame
        Dataframe containing non-filtered dataset
    train_size : float
        Fraction of train data
    block_number: int
        Number of block to plot temperature time series for
    """
    clear_output()
    plt.figure(figsize=(5, 5), dpi=120)
    plt.plot(range(1, int(len(heat_data) * train_size) + 1),
             heat_data.iloc[:int(len(heat_data) * train_size), block_number], label='train', color='black')

    plt.plot(range(int(len(heat_data) * train_size) + 1, len(heat_data) + 1),
             heat_data.iloc[int(len(heat_data) * train_size) :, block_number], label='validation', color='red')


    plt.ylabel('Temperature')
    plt.xlabel('Time tick')
    plt.title(f'Train validation data: block_number = {block_number}')
    plt.legend()
    plt.grid()
    plt.show()



def plot_losses(train_losses, test_losses, names=None):
    """
    Plots train and validation losses during training

    Parameters
    ----------
    heat_data : pd.DataFrame
        Dataframe containing non-filtered dataset
    train_size : float
        Fraction of train data
    block_number: int
        Number of block to plot temperature time series for
    """
    clear_output()
    plt.figure(figsize=(5, 5), dpi=120)

    if names is None:
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='train', color='black')
        plt.plot(range(1, len(test_losses) + 1), test_losses, label='validation', color='red')

    else:
        plt.plot(range(1, len(train_losses) + 1), train_losses, label=names[0])
        plt.plot(range(1, len(test_losses) + 1), test_losses, label=names[1])

    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.yscale('log')
    plt.title('Loss per epoch')
    plt.grid()

    plt.show()
