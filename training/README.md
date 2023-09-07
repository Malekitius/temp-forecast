Here you can find jupyter notebook for training recurrent neural network and pretrained model state (GRU).

## Model
You can choose rather GRU or LSTM to be trained, both are compatible with scripts, but GRU has a little higher performance.

## Filtration
You can activate filtration (smoothing) of training data in jupyter notebook. It will decrease loss during training but also decrease model ability to predict unfiltered data. Note that filtration smoothes temperature pikes what may be important if it is necessary to detect abrupt temperature increase.

## Optuna
Optuna code for tuning model hyperparameter is available in the notebook
