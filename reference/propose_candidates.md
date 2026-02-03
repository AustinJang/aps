# Propose New Candidate Points

Implements the epsilon-LCB acquisition strategy: exploitation (minimize
predicted value) and exploration (maximize uncertainty).

## Usage

``` r
propose_candidates(
  model_list,
  x_train,
  y_train,
  x_min,
  x_max,
  n_candidates,
  exploit_ratio = 0.5,
  n_starting = 20,
  tolerance = 0.01
)
```

## Arguments

- model_list:

  List of trained neural networks

- x_train:

  Current training data

- y_train:

  Current training targets

- x_min:

  Lower bound for parameters

- x_max:

  Upper bound for parameters

- n_candidates:

  Number of candidate points to propose

- exploit_ratio:

  Fraction of candidates for exploitation vs exploration (default 0.5)

- n_starting:

  Number of random starting points for optimization

- tolerance:

  Optimization tolerance

## Value

Matrix of proposed candidate points
