# Plot Performance Statistic Over Iterations

Visualizes how the performance statistic evolves across Bayesian
optimization iterations, distinguishing between exploration and
exploitation points.

## Usage

``` r
plot_performance(x, round_digits = 3, ...)
```

## Arguments

- x:

  An aps_result object

- round_digits:

  Number of decimal places for labels

- ...:

  Additional arguments (ignored)

## Value

A ggplot2 object

## Examples

``` r
if (FALSE) { # \dontrun{
result <- aps(obj_function, n_params = 2)
plot_performance(result)
} # }
```
