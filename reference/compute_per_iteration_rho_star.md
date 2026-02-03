# Compute Per-Iteration rho\* from Out-of-Sample Points

For each iteration, computes the maximum achievable correlation (rho\*)
using all new points evaluated in that iteration (the out-of-sample test
set). This matches how cor_out is computed in the main loop.

## Usage

``` r
compute_per_iteration_rho_star(result, k = 1)
```

## Arguments

- result:

  An aps_result object

- k:

  Number of nearest neighbors for noise estimation (default 1)

## Value

A data.frame with columns: iteration, rho_star, sigma2, tau2, n_points

## Details

Noise variance (sigma^2) is estimated using nearest neighbors from ALL
accumulated data (for density), while signal variance (tau^2) is
computed from the out-of-sample points' y variance.
