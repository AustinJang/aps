# Train a Single Neural Network

Trains a neural network on the provided data with early stopping.

## Usage

``` r
train_network(
  net,
  x_train,
  y_train,
  epochs = 1000,
  batch_size = 32,
  lr = 0.001,
  patience = 20,
  validation_split = 0.2,
  verbose = FALSE
)
```

## Arguments

- net:

  A torch nn_module

- x_train:

  Training features (matrix)

- y_train:

  Training targets (vector)

- epochs:

  Maximum number of epochs

- batch_size:

  Batch size for training

- lr:

  Learning rate

- patience:

  Early stopping patience

- validation_split:

  Fraction of data for validation

- verbose:

  Print progress

## Value

Trained network (modified in place)
