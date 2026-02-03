# aps: Adversarial Parameter Selection

R package for adversarial parameter selection in Monte Carlo simulation
studies. Uses Bayesian optimization with deep ensembles to identify
parameter configurations that expose worst-case method performance.

## Installation

``` r
# Install from GitHub
devtools::install_github("AustinJang/aps")

# torch is required - install if needed
install.packages("torch")
torch::install_torch()
```

## Quick Example

``` r
library(aps)

# Define an objective function that takes a parameter vector
# and returns a performance statistic (lower = worse)
obj_fn <- function(x) {
  # Example: compare two estimators on simulated data
  # x[1] = effect size, x[2] = noise level
  set.seed(42)
  y <- rnorm(100, mean = x[1], sd = x[2])

  # Performance: absolute bias of mean estimator
  abs(mean(y) - x[1])
}

# Run adversarial parameter selection
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

# Diagnostic plots (requires ggplot2)
plot(result, type = "performance")
plot(result, type = "correlation")
```

## Key Parameters

- `obj_function`: Function taking parameter vector, returning scalar
  performance
- `n_params`: Number of parameters in search space
- `x_min`, `x_max`: Parameter bounds
- `n_iter`: Number of Bayesian optimization iterations
- `n_obs`: Observations per iteration
- `exploit_ratio`: Balance between exploitation (0) and exploration (1),
  default 0.5

## Citation

If you use this package, please cite:

    Jang, A. (2025). Adversarial Parameter Selection for Monte Carlo Simulations.

------------------------------------------------------------------------

*This package was developed with the assistance of [Claude
Code](https://claude.ai/claude-code). All remaining errors are my own.*
