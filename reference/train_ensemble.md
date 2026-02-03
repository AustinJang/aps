# Train Deep Ensemble

Trains an ensemble of neural networks with varying architectures.

## Usage

``` r
train_ensemble(x_train, y_train, architecture_df, verbose = TRUE, ...)
```

## Arguments

- x_train:

  Training features (matrix)

- y_train:

  Training targets (vector)

- architecture_df:

  Data frame specifying architectures

- verbose:

  Print progress

- ...:

  Additional arguments passed to train_network

## Value

List of trained networks
