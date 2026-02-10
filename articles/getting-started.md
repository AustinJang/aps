# Getting Started with APS

## What is APS?

Adversarial Parameter Selection (APS) is an R package for finding
parameter configurations that expose worst-case performance in Monte
Carlo simulations. It uses Bayesian optimization with deep ensembles to
efficiently search the parameter space.

**Key use case**: You have a simulation study comparing statistical
methods. Instead of testing on a fixed grid of parameters, APS actively
searches for the parameters where your methods perform worst.

## Installation

``` r
# Install from GitHub
devtools::install_github("AustinJang/aps")

# torch is required - install if needed
install.packages("torch")
torch::install_torch()
```

## Basic Example

Define an objective function that takes a parameter vector and returns a
performance statistic (lower = worse):

``` r
library(aps)

# Simple example: find where an estimator has highest bias
obj_fn <- function(x) {
  # Simulate: x[1] = effect size, x[2] = noise level
  set.seed(42)  # For reproducibility within each call
  y <- rnorm(100, mean = x[1], sd = x[2])

  # Performance metric: negative absolute bias (lower = worse)
  -abs(mean(y) - x[1])
}

# Run APS
result <- aps(
  obj_function = obj_fn,
  n_params = 2,
  x_min = c(0, 0.1),
  x_max = c(1, 2),
  n_iter = 5,
  n_obs = 100
)

# View results
print(result)
summary(result)
```

## Understanding the Output

The result object contains:

- `best_x`: Parameter values producing the worst (lowest) performance
- `best_y`: The worst performance value found
- `x`, `y`: All evaluated parameter values and their performance
- `architecture_df`: Model performance across iterations

``` r
# Best adversarial point
result$best_x
result$best_y

# How many evaluations total?
nrow(result$x)

# Performance by iteration
tapply(result$y, result$iteration, min)
```

## Key Parameters

| Parameter        | Description                                     | Default  |
|------------------|-------------------------------------------------|----------|
| `n_params`       | Dimensions of parameter space                   | Required |
| `x_min`, `x_max` | Bounds (scalar or vector)                       | 0, 1     |
| `n_iter`         | Number of optimization iterations               | 10       |
| `n_obs`          | Evaluations per iteration                       | 500      |
| `epsilon`        | Exploration fraction (0 = exploit, 1 = explore) | 0.5      |
| `early_stop`     | Enable convergence-based stopping               | FALSE    |

## Diagnostic Plots

APS provides two diagnostic plots (requires ggplot2):

``` r
# Performance over iterations
plot(result, type = "performance")

# Model correlation with noise-adjusted ceiling
plot(result, type = "correlation")
```

The correlation plot shows: - Circles: in-sample correlation (training
fit) - Squares: out-of-sample correlation (predictive quality) - Green
dashed line: rho\* (maximum achievable given noise)

## Early Stopping

Enable automatic stopping when the surrogate has learned all learnable
signal:

``` r
result <- aps(
  obj_function = obj_fn,
  n_params = 2,
  n_iter = 20,
  n_obs = 100,
  early_stop = TRUE,
  rho_tilde_threshold = 0.95,
  min_stable_iters = 2
)

# Did it stop early?
result$stopped_early
result$final_iteration
```

## Checkpointing

For long runs, save progress after each iteration:

``` r
result <- aps(
  obj_function = expensive_simulation,
  n_params = 5,
  n_iter = 50,
  checkpoint_file = "aps_checkpoint.rds"
)

# Resume if interrupted
result <- aps(
  obj_function = expensive_simulation,
  n_params = 5,
  n_iter = 50,
  resume_from = "aps_checkpoint.rds"
)
```

## Next Steps

- See
  [`vignette("vector-mode")`](https://austinjang.github.io/aps/articles/vector-mode.md)
  for multi-output simulations with deferred scalarization
- See
  [`vignette("diagnostics")`](https://austinjang.github.io/aps/articles/diagnostics.md)
  for understanding convergence diagnostics
