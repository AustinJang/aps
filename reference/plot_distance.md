# Plot Exploitation Point Distance Tracking

Visualizes cumulative distance traveled by exploitation points across
the optimization. Each exploitation point is shown on the x-axis, with
the y-axis showing cumulative distance from the starting point.

## Usage

``` r
plot_distance(x, normalize = TRUE, ...)
```

## Arguments

- x:

  An aps_result object

- normalize:

  Logical. If TRUE, distances are normalized by the parameter space
  diameter (max possible distance). Default TRUE.

- ...:

  Additional arguments (ignored)

## Value

A ggplot2 object showing cumulative distance traveled

## Details

This diagnostic complements rho_tilde by showing WHERE the surrogate
thinks the minimum is, not just how well it predicts. Key patterns:

- Steep slope: surrogate moving through parameter space (learning)

- Flat regions: surrogate staying in same area (converged or stuck)

- Jumps at iteration boundaries: new models found different region

- Shaky within iteration: models disagree about minimum location

- Smooth within iteration: models agree on minimum location

## Examples

``` r
if (FALSE) { # \dontrun{
result <- aps(obj_function, n_params = 2)
plot_distance(result)
} # }
```
