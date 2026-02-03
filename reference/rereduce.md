# Re-reduce an APS result with a new reduction function

Given an existing aps_result with stored raw outputs (`y_raw`),
recompute the scalar `y` values using a new reduction function. This
allows exploring different scalarizations without re-running expensive
simulations.

## Usage

``` r
rereduce(result, reduction, invalid_penalty = NULL)
```

## Arguments

- result:

  An aps_result object with non-NULL `y_raw`

- reduction:

  A function mapping a named numeric vector to a scalar

- invalid_penalty:

  Optional penalty value for invalid y values (NA, NULL, Inf, NaN,
  non-numeric). If NULL (default), uses max(valid_y) + 2\*sd(valid_y).

## Value

A modified aps_result with updated `y`, `best_x`, `best_y`, `reduction`,
and `invalid_idx`
