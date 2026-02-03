# Compute noise-adjusted convergence diagnostic (reliability ratio)

Computes the reliability ratio, which adjusts the raw out-of-sample
correlation for irreducible noise. A perfect surrogate achieves
\\\rho^\* = \tau / \sqrt{\tau^2 + \sigma^2}\\, not 1, due to Monte Carlo
noise. The reliability ratio \\\tilde{\rho} = \rho\_{out} / \rho^\*\\
indicates how close the surrogate is to this theoretical maximum.

## Usage

``` r
reliability_ratio(result, k = 1)
```

## Arguments

- result:

  An aps_result object

- k:

  Number of nearest neighbors to use for variance estimation (default
  1). Larger k reduces variance but increases bias.

## Value

A list with class "aps_reliability" containing:

- sigma2:

  Estimated noise variance (average MC noise)

- tau2:

  Estimated signal variance

- rho_star:

  Maximum achievable correlation given noise level

- rho_out:

  Best out-of-sample correlation from the final iteration

- rho_tilde:

  Reliability ratio: rho_out / rho_star

- snr:

  Signal-to-noise ratio: tau2 / sigma2

- interpretation:

  A text interpretation of the diagnostic

## Details

Noise variance \\\sigma^2\\ is estimated using nearest-neighbor
differences (Gasser et al. 1986), which does not require repeated
evaluations at the same point.

The diagnostic answers the question: "Is my surrogate as good as it can
be given the noise level?" If \\\tilde{\rho} \approx 1\\, the surrogate
has extracted all learnable signal. If \\\tilde{\rho} \ll 1\\, there is
room for improvement (more data, better architecture, etc.).

The noise variance is estimated using the nearest-neighbor difference
estimator: for each point, we find its nearest neighbor in parameter
space and compute the squared difference in y values. Under smoothness
assumptions, this differences out the signal, leaving only noise.

## References

Gasser, T., Sroka, L., & Jennen-Steinmetz, C. (1986). Residual variance
and residual pattern in nonlinear regression. Biometrika, 73(3),
625-633.

## Examples

``` r
if (FALSE) { # \dontrun{
result <- aps(obj_function = my_sim, n_params = 5, ...)
diag <- reliability_ratio(result)
print(diag)
} # }
```
