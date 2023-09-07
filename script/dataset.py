# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def scale_temperature(x, min_temperature=10, max_temperature=70):
    """
    Returns scaled temperature time series

    Parameters
    ----------
    x : pd.DataFrame
        Dataframe containing time series to be scaled
    min_temperature : float
        Minimal temperature that will be scaled as 0
    max_temperature : float
        Maximal temperature that will be scaled as 1

    Returns
    ----------
    x_scaled : pd.DataFrame
        Scaled temperature time series
    """
    x_scaled = (x - min_temperature) / (max_temperature - min_temperature)
    return x_scaled



def unscale_temperature(x, min_temperature=10, max_temperature=70):
    """
    Returns unscaled temperature time series

    Parameters
    ----------
    x : pd.DataFrame
        Dataframe containing time series to be unscaled
    min_temperature : float
        Minimal temperature that is scaled as 0
    max_temperature : float
        Maximal temperature that is scaled as 1

    Returns
    ----------
    x_unscaled : pd.DataFrame
        Unscaled temperature time series
    """
    x_unscaled = x * (max_temperature - min_temperature) + min_temperature
    return x_unscaled



class Heat_Sequence_Dataset(Dataset):
    """
    Class for making nn.Dataset from temperature time series data
    """
    def __init__(self, heat_data, seq_len=64):
        """
        Initializes instance of class Heat_Sequence_Dataset

        Parameters
        ----------
        heat_data : pd.DataFrame
            Dataframe containing time series
        seq_len : int
            Number of consequent temperature values that are simultaneously input into model
        """
        self.heat_data = heat_data
        self.seq_len = seq_len

        self.heat_data.iloc[:, :-2] = scale_temperature(self.heat_data.iloc[:, :-2])

    def __len__(self):
        """
        Returns length of the dataset
        """
        return len(self.heat_data) - self.seq_len - 1

    def __getitem__(self, item):
        """
        Returns pair of input (sequence of vectors) and output (one vector) temperature values

        Parameters
        ----------
        item : int
            Index of sequence to be returned

        Returns
        ----------
        x : torch.tensor
            Input sequence of vectors
        y : torch.tensor
            Output vector
    """
        x = torch.tensor(self.heat_data.iloc[item: item + self.seq_len, :].values, dtype=torch.float32)
        y = torch.tensor(self.heat_data.iloc[item + self.seq_len, :-2].values, dtype=torch.float32)
        return x, y