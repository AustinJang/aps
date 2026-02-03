# Train a Single Model (Helper for Parallelization)

Creates and trains a single neural network. Used internally for parallel
training.

## Usage

``` r
train_single_model(arch_row, x_train, y_train, n_params, ...)
```

## Arguments

- arch_row:

  Single row from architecture data frame

- x_train:

  Training features

- y_train:

  Training targets

- n_params:

  Number of input parameters

- ...:

  Additional arguments passed to train_network

## Value

Trained network
