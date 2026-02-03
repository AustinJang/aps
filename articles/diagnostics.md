# Understanding Convergence Diagnostics

## The Core Question

After running APS, how do you know if the search has converged? A raw
out-of-sample correlation of 0.6 could mean:

1.  **Bad surrogate**: The model is underfitting; more iterations would
    help
2.  **Noisy surface**: The simulation has high Monte Carlo noise; no
    model can do better

APS provides diagnostics to distinguish these cases.

## The Reliability Ratio

The key insight is that Monte Carlo noise creates a **ceiling** on
achievable correlation.

If your simulation is $y = m(\mathbf{x}) + \varepsilon$ where: -
$m(\mathbf{x})$ is the true response surface - $\varepsilon$ is MC noise
with variance $\sigma^{2}$ - Signal variance is
$\tau^{2} = \text{Var}\left( m(\mathbf{x}) \right)$

Then even a **perfect** surrogate achieves at most:
$$\rho^{*} = \frac{\tau}{\sqrt{\tau^{2} + \sigma^{2}}}$$

The **reliability ratio** normalizes observed correlation by this
ceiling: $$\widetilde{\rho} = \frac{\rho_{\text{out}}}{\rho^{*}}$$

- $\widetilde{\rho} \approx 1$: Surrogate has learned all learnable
  signal
- $\widetilde{\rho} \ll 1$: Room for improvement

## Using reliability_ratio()

``` r
library(aps)

# Run APS
result <- aps(
  obj_function = my_simulation,
  n_params = 3,
  n_iter = 10,
  n_obs = 100
)

# Get the diagnostic
diag <- reliability_ratio(result)
print(diag)
```

Output:

    APS Convergence Diagnostic: Reliability Ratio
    ==============================================

    Sample size: 1100 points
    Nearest neighbors used: k = 1

    Variance Decomposition:
      Total variance (Var(y)):  2.4500
      Noise variance (sigma^2): 0.3200
      Signal variance (tau^2):  2.1300
      Signal-to-noise ratio:    6.66

    Correlation Diagnostic:
      Maximum achievable (rho*):   0.932
      Observed out-of-sample:      0.891
      Reliability ratio (rho~):    0.956

    Interpretation:
      Excellent (rho_tilde = 0.96). The surrogate has extracted nearly all
      learnable signal. Continued optimization is unlikely to find
      substantially better regions.

## The Correlation Plot

``` r
plot(result, type = "correlation")
```

This plot shows:

- **Circles**: In-sample correlation (training fit)
- **Squares**: Out-of-sample correlation (predictive ability)
- **Arrows**: Change from in-sample to out-of-sample
- **Green dashed lines**: Per-iteration $\rho^{*}$ (ceiling)

### Reading the Plot

**Good convergence**: - Out-of-sample correlations (squares) near or at
the green line - Stable across iterations

**Overfitting**: - Large gaps between circles and squares - In-sample
high, out-of-sample low

**Underfitting**: - Both in-sample and out-of-sample well below the
green line - May need more data or different architectures

## Per-Iteration rho\*

The green segments can **vary across iterations**. This happens when:

1.  **Heteroskedastic noise**: Different regions have different noise
    levels
2.  **Exploitation focusing**: Later iterations sample noisier regions
    near the minimum

``` r
# Example with location-dependent noise
hetero_fn <- function(x) {
  signal <- sum(x^2)
  # Noise is HIGH near the minimum, LOW in corners
  local_sigma <- 0.3 + 2 * exp(-signal)
  signal + rnorm(1, sd = local_sigma)
}

result <- aps(
  obj_function = hetero_fn,
  n_params = 2,
  x_min = -2, x_max = 2,
  n_iter = 5,
  n_obs = 80
)

plot(result, type = "correlation")
# Green segments will be LOWER for later iterations
# (which focus on the noisy minimum region)
```

## Noise Estimation

APS estimates $\sigma^{2}$ using nearest-neighbor differences:

$${\widehat{\sigma}}^{2} = \frac{1}{2n}\sum\limits_{i = 1}^{n}\left( y_{i} - y_{nn{(i)}} \right)^{2}$$

This works because nearby points have similar signal, so differencing
isolates noise.

You can adjust the number of neighbors:

``` r
# Use k=3 neighbors for smoother estimate (more bias, less variance)
diag <- reliability_ratio(result, k = 3)

# Also works in plots
plot(result, type = "correlation", k = 3)
```

## Early Stopping

Use the reliability ratio for automatic stopping:

``` r
result <- aps(
  obj_function = my_simulation,
  n_params = 5,
  n_iter = 50,  # Maximum iterations
  n_obs = 100,
  early_stop = TRUE,
  rho_tilde_threshold = 0.95,  # Stop when 95% of signal learned

min_stable_iters = 2          # AND minimum stable for 2 iterations
)

# Check if stopped early
result$stopped_early
result$final_iteration
```

The stopping condition requires **both**: 1. $\widetilde{\rho} \geq$
threshold (surrogate is good) 2. Minimum hasn’t improved for N
iterations (nothing better to find)

## Signal-to-Noise Ratio

The diagnostic also reports SNR = $\tau^{2}/\sigma^{2}$:

- **SNR \> 10**: Strong signal, noise is minor
- **SNR ≈ 1**: Signal and noise comparable
- **SNR \< 1**: Noise dominates; may need more replications per
  evaluation

``` r
diag <- reliability_ratio(result)
diag$snr

# If SNR is very low, consider:
# 1. Increasing replications within your simulation
# 2. Using variance reduction techniques
# 3. Accepting that this surface is inherently noisy
```

## Summary

| Diagnostic                      | Meaning         | Action                    |
|---------------------------------|-----------------|---------------------------|
| $\widetilde{\rho} \approx 1$    | Converged       | Stop, analyze results     |
| $\widetilde{\rho} < 0.7$        | Underfitting    | More iterations or data   |
| SNR \< 1                        | Noise-dominated | Increase replications     |
| Per-iteration $\rho^{*}$ varies | Heteroskedastic | Normal; interpret locally |
