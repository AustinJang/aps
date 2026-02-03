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
  y_raw_init = NULL,
  num_cores = 1,
  exploit_ratio = 0.5,
  competitiveness_param = 0.2,
  tolerance = 0.01,
  verbose = TRUE,
  checkpoint_file = NULL,
  resume_from = NULL,
  use_genetic = FALSE,
  reduction = NULL,
  output_names = NULL,
  architecture_config = NULL,
  early_stop = FALSE,
  rho_tilde_threshold = 0.95,
  min_stable_iters = 2,
  invalid_penalty = NULL
)
```

## Arguments

- obj_function:

  Function that takes a parameter vector and returns either:

  - A scalar performance statistic (classic usage), or

  - A named numeric vector of intermediate outputs (vector mode).

  In vector mode, a `reduction` function must be provided to map the
  output vector to a scalar. Lower values indicate worse performance.

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

  Optional vector of initial reduced performance statistics. For vector
  mode, you can also pass `y_raw_init` instead.

- y_raw_init:

  Optional matrix of initial raw objective outputs (vector mode only).
  One row per observation, one column per output component. If provided
  with a `reduction`, `y_init` is computed automatically.

- num_cores:

  Number of cores for parallel evaluation (default 1)

- exploit_ratio:

  Fraction of candidates for exploitation vs exploration (default 0.5)

- competitiveness_param:

  Threshold for eliminating underperforming models (default 0.2)

- tolerance:

  Optimization convergence tolerance (default 0.01)

- verbose:

  Print progress messages (default TRUE)

- checkpoint_file:

  Optional file path to save checkpoints after each iteration. Allows
  resuming if the process is interrupted.

- resume_from:

  Optional. Either a file path to a checkpoint or an aps_result object
  to resume from. When resuming, iterations continue from where they
  left off. If the `reduction` function has changed since the previous
  run, the stored raw outputs (`y_raw`) are re-scalarized without
  re-running simulations. The surrogate is then retrained on the new
  scalar targets.

- use_genetic:

  Logical. If TRUE, use genetic algorithm-style mutations to generate
  new architectures from well-performing parents. If FALSE (default),
  use purely random architecture generation.

- reduction:

  Optional function that maps a named numeric vector (one row of
  objective output) to a scalar. Required when `obj_function` returns
  vectors. Can be changed on resume to re-scalarize existing data
  without re-running simulations. Examples: `function(v) -max(v)`,
  `function(v) v["bias_A"] - v["bias_B"]`. If NULL and `obj_function`
  returns a scalar, behaves identically to the original package (full
  backward compatibility).

- output_names:

  Optional character vector naming the outputs of `obj_function`. If
  NULL, names are inferred from the first evaluation. Only relevant in
  vector mode.

- architecture_config:

  Optional list specifying neural network architecture parameter ranges
  for the ensemble. If NULL, uses defaults. Supported keys:

  - `depth`: Integer vector of allowed depths (default: 1:5)

  - `width`: Integer vector of allowed widths (default: c(32, 64, 128,
    256, 512))

  - `reg`: Numeric vector of allowed L2 regularization values (default:
    seq(0, 0.0005, 0.0001))

  - `dropout`: Numeric vector of allowed dropout values (default: seq(0,
    0.5, 0.1))

  - `activation`: Character vector of allowed activations (default:
    c("relu", "sigmoid", "tanh"))

  Example:
  `architecture_config = list(depth = 1:3, activation = c("relu", "tanh"))`

- early_stop:

  Logical. If TRUE, stop early when the surrogate has learned all
  learnable signal (rho_tilde \>= threshold) AND the minimum has been
  stable for `min_stable_iters` iterations. Default FALSE.

- rho_tilde_threshold:

  Threshold for the reliability ratio (rho_out / rho\*) above which the
  surrogate is considered to have learned all signal. Default 0.95.

- min_stable_iters:

  Number of consecutive iterations without improvement in the minimum
  before early stopping is triggered. Default 2.

- invalid_penalty:

  Optional penalty value for invalid y values (NA, NULL, Inf, NaN,
  non-numeric). When the reduction function returns an invalid value,
  APS assigns this penalty and tracks the index. If NULL (default), uses
  max(valid_y) + 2\*sd(valid_y) to steer exploration away from crash
  regions.

## Value

A list with class "aps_result" containing:

- x:

  Matrix of all evaluated parameter values

- y:

  Vector of all (reduced) performance statistics

- y_raw:

  Matrix of raw objective outputs (NULL in scalar mode). Rows =
  evaluations, columns = output components.

- architecture_df:

  Data frame of model architectures and performance

- best_x:

  Parameter values with lowest (worst) performance

- best_y:

  Lowest performance statistic found

- iteration:

  Vector indicating which iteration each point was from

- reduction:

  The reduction function used (NULL in scalar mode)

- output_names:

  Names of the output vector components (NULL in scalar mode)

- invalid_idx:

  Integer vector of indices where y was invalid and penalty applied

- stopped_early:

  Logical indicating if early stopping was triggered

- final_iteration:

  The last iteration completed (may be less than n_iter if stopped
  early)

## Examples

``` r
if (FALSE) { # \dontrun{
# Scalar mode (classic, fully backward-compatible):
obj_fn <- function(x) sin(x[1]) * cos(x[2])

result <- aps(
  obj_function = obj_fn,
  n_params = 2,
  x_min = 0,
  x_max = 2 * pi,
  n_iter = 5,
  n_obs = 100
)

# Vector mode: objective returns multiple metrics, reduce to scalar
obj_fn_vec <- function(x) c(bias_A = x[1]^2, bias_B = (x[2]-1)^2)

result <- aps(
  obj_function = obj_fn_vec,
  n_params = 2,
  x_min = -2,
  x_max = 2,
  n_iter = 5,
  n_obs = 100,
  reduction = function(v) v["bias_A"] - v["bias_B"]
)

# Resume with a different reduction — no re-simulation needed:
result2 <- aps(
  obj_function = obj_fn_vec,
  n_params = 2,
  x_min = -2,
  x_max = 2,
  n_iter = 10,
  n_obs = 100,
  reduction = function(v) -v["bias_B"],
  resume_from = result
)
} # }
```
