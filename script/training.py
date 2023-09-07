# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import optuna
from tqdm.notebook import tqdm

from plotting import plot_losses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_test_loss(model, criterion, test_loader):
    """
    Calculates loss on validation/test dataset using specified criterion

    Parameters
    ----------
    model : model object
        Model to return predicted values
    criterion : loss function instance
    test_loader : torch.utils.data.DataLoader
        Validation/test dataset dataloader.

    Returns
    ----------
    test_loss : float
        Loss calculated between true and predicted values
    """
    model.eval()
    test_loss = 0

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            prediction = model(x)
            loss = criterion(prediction, y)

        test_loss += loss.item() * x.shape[0]

    return test_loss



def train(model, criterion, optimizer, scheduler,
          train_loader, val_loader, num_epochs, epoch_freq,
          plot_progress, print_progress, trial=None):
    """
    Trains neural model

    Parameters
    ----------
    model : model instance
        Model that is being trained
    criterion : loss function instance
    optimizer : optimizer instance
    scheduler : scheduler instance
    train_loader : torch.utils.data.DataLoader
        Train dataset dataloader
    val_loader : torch.utils.data.DataLoader
        Validation dataset dataloader
    num_epochs : int
        Number of training epochs (default 100)
    epoch_freq : int
        Number of epochs between two plotting/printing events
    plot_progress : bool
        If True, activates plotting function every `epoch_freq` epoch
    print_progress : bool
        If True, prints train, val losses every 'epoch_freq' epoch
    trial : optuna.trial.Trial instance
        If not None, initializes optuna-based hyperparameter tuning including pruning unsuccessful tries (default None)
    """

    train_losses, val_losses = [], []

    for epoch in tqdm(range(1, num_epochs + 1)):
        train_loss = 0.0
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            # x: batch_size x seq_len x input_size
            prediction = model(x)
            # prediction: batch_size x output_size
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.shape[0]


        train_loss /= len(train_loader.dataset)
        train_losses += [train_loss]

        val_loss = calculate_test_loss(model, criterion, val_loader)

        if scheduler is not None:
            scheduler.step(val_loss)

        val_loss /= len(val_loader.dataset)
        val_losses += [val_loss]

        if trial is not None:
            # Early-stopping
            trial.report(val_loss, epoch)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if epoch % epoch_freq == 0:
            if plot_progress:
                plot_losses(train_losses, val_losses)
            if print_progress:
                print(f"Epoch: {epoch},".ljust(15), "loss train: %1.5f" % (train_loss), "loss test: %1.5f" % (val_loss))