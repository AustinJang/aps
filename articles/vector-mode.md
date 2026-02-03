# Vector Mode and Deferred Scalarization

## The Problem

Real simulations return multiple statistics: bias for method A, bias for
method B, RMSE, coverage, etc. After running an expensive search, you
often want to ask a different question:

- “Where does A fail?” → “Where does A fail *relative to B*?”
- “Maximize RMSE” → “Maximize coverage gap”

With scalar mode, you’d have to re-run everything. **Vector mode**
solves this.

## Vector Mode Basics

Instead of returning a scalar, your objective function returns a named
vector:

``` r
library(aps)

# Returns a vector of metrics
obj_fn <- function(x) {
  # Simulate data
  set.seed(42)
  y <- rnorm(100, mean = x[1], sd = x[2])

  c(
    bias_mean = mean(y) - x[1],
    bias_median = median(y) - x[1],
    rmse_mean = sqrt(mean((y - x[1])^2)),
    coverage = mean(abs(y - x[1]) < 1.96 * x[2])
  )
}

# The reduction function maps vector → scalar
# APS minimizes this scalar
result <- aps(
  obj_function = obj_fn,
  n_params = 2,
  x_min = c(0, 0.1),
  x_max = c(1, 2),
  n_iter = 5,
  n_obs = 100,
  reduction = function(v) -abs(v["bias_mean"])  # Minimize negative bias
)
```

## Deferred Scalarization with rereduce()

The magic: APS stores the full vector at every point. Change the
question without re-running:

``` r
# Original question: where is bias_mean worst?
result$best_y
result$best_x

# New question: where is coverage lowest?
result_coverage <- rereduce(result, function(v) v["coverage"])
result_coverage$best_y
result_coverage$best_x

# Another question: where does mean beat median?
result_gap <- rereduce(result, function(v) {
  abs(v["bias_mean"]) - abs(v["bias_median"])  # Positive = mean worse
})
result_gap$best_y
result_gap$best_x
```

## Accessing Raw Outputs

The full output matrix is stored in `y_raw`:

``` r
# Raw outputs: rows = evaluations, columns = output components
dim(result$y_raw)
colnames(result$y_raw)

# Extract all coverage values
result$y_raw[, "coverage"]

# Find which parameters gave coverage < 0.9
low_coverage_idx <- which(result$y_raw[, "coverage"] < 0.9)
result$x[low_coverage_idx, ]
```

## Real-World Example: Comparing Estimators

A typical use case: comparing DID estimators across effect trajectories.

``` r
# Objective returns per-estimator RMSE
did_sim <- function(alpha) {
  # alpha is a 5-element effect trajectory
  # ... run simulation ...

  c(
    twfe_rmse = 0.5,      # Placeholder values
    csa_rmse = 0.3,
    gardner_rmse = 0.4,
    ascm_rmse = 0.2
  )
}

# Find where ANY estimator fails
result <- aps(
  obj_function = did_sim,
  n_params = 5,
  x_min = -1, x_max = 1,
  n_iter = 10,
  n_obs = 50,
  reduction = function(v) -max(v)  # Worst across all estimators
)

# Post-hoc: where does TWFE specifically fail?
result_twfe <- rereduce(result, function(v) -v["twfe_rmse"])

# Where is the gap between TWFE and ASCM largest?
result_gap <- rereduce(result, function(v) {
  v["twfe_rmse"] - v["ascm_rmse"]  # Positive = TWFE worse
})
```

## Resume with Different Reduction

You can resume an existing run with a new reduction function:

``` r
# Original run
result1 <- aps(
  obj_function = obj_fn,
  n_params = 2,
  n_iter = 5,
  reduction = function(v) -v["bias_mean"],
  checkpoint_file = "checkpoint.rds"
)

# Resume with different reduction - no re-simulation!
# Stored y_raw is re-scalarized with new reduction
result2 <- aps(
  obj_function = obj_fn,
  n_params = 2,
  n_iter = 10,
  reduction = function(v) -v["coverage"],  # New question
  resume_from = "checkpoint.rds"
)
```

## Summary Statistics

The summary function shows statistics for each output component:

``` r
summary(result)
# Output mode: vector (4 components)
# Output names: bias_mean, bias_median, rmse_mean, coverage
#
# Raw Output Statistics (across all evaluations):
#   bias_mean: min=-0.12, mean=0.01, max=0.15
#   ...
#
# Reduced Performance Statistics:
#   Min (adversarial): -0.12
#   ...
```

## Key Points

1.  **Store once, ask many questions**: The expensive simulation runs
    once; reductions are cheap
2.  **Named vectors**: Use names for clarity and to reference outputs in
    reduction functions
3.  **rereduce() is instant**: No simulation, just re-computes y from
    y_raw
4.  **Resume preserves y_raw**: Even when resuming with a new reduction
