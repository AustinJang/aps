# Adversarial Parameter Selection

Main function for adversarial parameter selection using Bayesian
optimization with deep ensembles. Iteratively trains an ensemble of
neural networks to approximate the objective function, then uses the
ensemble to select parameter values likely to produce poor method
performance.

## Usage

``` r
aps(
  obj_function,
  n_params,
  x_min = 0,
  x_max = 1,
  n_iter = 10,
  n_obs = 500,
  n_models = 5,
  x_init = NULL,
  y_init = NULL,
  num_cores = 1,
  competitiveness_param = 0.2,
  tolerance = 0.01,
  verbose = TRUE
)
```

## Arguments

- obj_function:

  Function that takes a parameter vector and returns a scalar
  performance statistic. Lower values indicate worse performance
  (adversarial).

- n_params:

  Number of parameters in the search space

- x_min:

  Lower bound for all parameters (scalar or vector)

- x_max:

  Upper bound for all parameters (scalar or vector)

- n_iter:

  Number of Bayesian optimization iterations

- n_obs:

  Number of new observations per iteration

- n_models:

  Number of models in the ensemble

- x_init:

  Optional matrix of initial parameter values

- y_init:

  Optional vector of initial performance statistics

- num_cores:

  Number of cores for parallel evaluation (default 1)

- competitiveness_param:

  Threshold for eliminating underperforming models (default 0.2)

- tolerance:

  Optimization convergence tolerance (default 0.01)

- verbose:

  Print progress messages (default TRUE)

## Value

A list with class "aps_result" containing:

- x:

  Matrix of all evaluated parameter values

- y:

  Vector of all performance statistics

- architecture_df:

  Data frame of model architectures and performance

- best_x:

  Parameter values with lowest (worst) performance

- best_y:

  Lowest performance statistic found

- iteration:

  Vector indicating which iteration each point was from

## Examples

``` r
if (FALSE) { # \dontrun{
# Simple example: find parameters where sin function is minimized
obj_fn <- function(x) sin(x[1]) * cos(x[2])

result <- aps(
  obj_function = obj_fn,
  n_params = 2,
  x_min = 0,
  x_max = 2 * pi,
  n_iter = 5,
  n_obs = 100
)

print(result$best_x)
print(result$best_y)
} # }
```
