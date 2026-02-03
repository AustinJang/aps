# Train Deep Ensemble

Trains an ensemble of neural networks with varying architectures.
Supports parallel training across multiple CPU cores.

## Usage

``` r
train_ensemble(
  x_train,
  y_train,
  architecture_df,
  num_cores = 1,
  verbose = TRUE,
  ...
)
```

## Arguments

- x_train:

  Training features (matrix)

- y_train:

  Training targets (vector)

- architecture_df:

  Data frame specifying architectures

- num_cores:

  Number of CPU cores for parallel training (default 1 = sequential)

- verbose:

  Print progress

- ...:

  Additional arguments passed to train_network

## Value

List of trained networks
