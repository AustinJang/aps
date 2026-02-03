# Generate Random Architecture

Generates a random neural network architecture specification with
uniform width across all hidden layers. This is a reasonable default for
most use cases.

## Usage

``` r
generate_architectures(n, config = NULL)
```

## Arguments

- n:

  Number of architectures to generate

- config:

  Optional list specifying architecture parameter ranges. If NULL, uses
  defaults. Supported keys:

  - `depth`: Integer vector of allowed depths (default: 1:5)

  - `width`: Integer vector of allowed widths (default: c(32, 64, 128,
    256, 512))

  - `reg`: Numeric vector of allowed regularization values (default:
    seq(0, 0.0005, 0.0001))

  - `dropout`: Numeric vector of allowed dropout values (default: seq(0,
    0.5, 0.1))

  - `activation`: Character vector of allowed activations (default:
    c("relu", "sigmoid", "tanh"))

## Value

A data frame with architecture specifications

## Details

For custom architectures (e.g., funnel/bottleneck shapes, per-layer
activations), users can either: (1) provide their own architecture_df to
aps(), or (2) modify this function to generate different architecture
specifications. The required columns are: depth, width, reg, dropout,
activation.
