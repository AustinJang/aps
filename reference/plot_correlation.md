# Plot Model Correlation Diagnostics

Visualizes in-sample and out-of-sample correlations for each model
across iterations. Arrows show the change from in-sample to
out-of-sample performance.

## Usage

``` r
plot_correlation(x, competitiveness = 0.2, ...)
```

## Arguments

- x:

  An aps_result object

- competitiveness:

  Threshold for highlighting "good" models (default 0.2)

- ...:

  Additional arguments (ignored)

## Value

A ggplot2 object

## Examples

``` r
if (FALSE) { # \dontrun{
result <- aps(obj_function, n_params = 2)
plot_correlation(result)
} # }
```
