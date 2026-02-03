# Find Minimum via Gradient Descent

Uses gradient descent to minimize a function (typically the ensemble
prediction or negative uncertainty).

## Usage

``` r
find_minimum(
  f_approx,
  starting_points,
  x_min,
  x_max,
  lr = 0.1,
  max_iter = 500,
  tolerance = 0.01
)
```

## Arguments

- f_approx:

  A function that takes a torch tensor and returns predictions

- starting_points:

  Matrix of starting points (each row is a point)

- x_min:

  Lower bound for parameters

- x_max:

  Upper bound for parameters

- lr:

  Learning rate

- max_iter:

  Maximum iterations

- tolerance:

  Convergence tolerance

## Value

List with optimized x values and corresponding y values
