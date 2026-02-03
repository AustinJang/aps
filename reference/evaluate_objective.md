# Evaluate Objective Function

Helper function to evaluate the objective function, optionally in
parallel. Handles both scalar and vector-valued objective functions.

## Usage

``` r
evaluate_objective(
  obj_function,
  x,
  num_cores = 1,
  reduction = NULL,
  output_names = NULL,
  verbose = FALSE,
  progress_prefix = NULL
)
```

## Arguments

- obj_function:

  The objective function

- x:

  Matrix of parameter values

- num_cores:

  Number of cores

- reduction:

  Optional reduction function for vector outputs

- output_names:

  Optional names for output components

- verbose:

  Print progress (default FALSE)

- progress_prefix:

  Optional prefix for progress messages (e.g., "Initial sample")

## Value

List with:

- y:

  Vector of (reduced) scalar performance statistics

- y_raw:

  Matrix of raw outputs (NULL if obj_function returns scalars)

- output_names:

  Names of output components (NULL if scalar)
