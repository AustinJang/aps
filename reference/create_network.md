# Create a Neural Network Module

Creates a feedforward neural network with specified architecture.

## Usage

``` r
create_network(input_dim, width, depth, activation = "relu", dropout = 0)
```

## Arguments

- input_dim:

  Number of input features (parameters)

- width:

  Number of units in hidden layers

- depth:

  Number of hidden layers

- activation:

  Activation function ("relu", "sigmoid", or "tanh")

- dropout:

  Dropout rate (0 to 1)

## Value

A torch nn_module
