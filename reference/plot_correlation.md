# Plot Model Correlation Diagnostics

Visualizes in-sample and out-of-sample correlations for each model
across iterations. Arrows show the change from in-sample to
out-of-sample performance. Optionally shows the maximum achievable
correlation (rho\*) given Monte Carlo noise, computed per-iteration from
all out-of-sample points.

## Usage

``` r
plot_correlation(x, competitiveness = 0.2, show_rho_star = TRUE, k = 1, ...)
```

## Arguments

- x:

  An aps_result object

- competitiveness:

  Threshold for highlighting "good" models (default 0.2)

- show_rho_star:

  Logical. If TRUE, compute and display the maximum achievable
  correlation given noise level, per iteration. Default TRUE.

- k:

  Number of nearest neighbors for noise estimation. Default 1.

- ...:

  Additional arguments (ignored)

## Value

A ggplot2 object

## Details

The green dashed segments show rho\* for each iteration block, computed
from the same points used to calculate cor_out. rho\* may vary across
iterations as APS samples different regions with different noise levels.

## Examples

``` r
if (FALSE) { # \dontrun{
result <- aps(obj_function, n_params = 2)
plot_correlation(result)
plot_correlation(result, show_rho_star = FALSE)  # without noise adjustment
} # }
```
