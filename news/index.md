# Changelog

## aps 0.1.0

First release of the Adversarial Parameter Selection package.

### Features

#### Core APS Algorithm

- Neural network ensemble surrogates for efficient parameter space
  exploration
- Exploitation/exploration balance via `exploit_ratio`
- Competitive model selection: underperforming architectures replaced
  each iteration
- Configurable network architectures via `architecture_config`

#### Vector Mode and Deferred Scalarization

- Objective functions can return named vectors of metrics
- `reduction` function maps vector outputs to scalar optimization target
- [`rereduce()`](https://austinjang.github.io/aps/reference/rereduce.md)
  allows post-hoc re-scalarization without re-running simulations
- Full raw outputs stored in `y_raw` for post-hoc analysis

#### Checkpointing and Resume

- Save progress after each iteration via `checkpoint_file`
- Auto-resume from existing checkpoint (no `resume_from` needed if
  checkpoint exists)
- Resume with different `reduction` function to optimize new target

#### Early Stopping

- Stop when surrogate has learned all signal: `early_stop = TRUE`
- Configurable thresholds: `rho_tilde_threshold`, `min_stable_iters`

#### Invalid Value Handling

- Graceful handling of NA, Inf, NaN from reduction functions
- Configurable penalty via `invalid_penalty`
- Crash indices tracked in `result$invalid_idx`

#### Diagnostics

- [`reliability_ratio()`](https://austinjang.github.io/aps/reference/reliability_ratio.md):
  noise-adjusted convergence diagnostic
- [`plot_performance()`](https://austinjang.github.io/aps/reference/plot_performance.md):
  objective values over iterations
- [`plot_correlation()`](https://austinjang.github.io/aps/reference/plot_correlation.md):
  surrogate model quality with rho\* ceiling
- [`plot_distance()`](https://austinjang.github.io/aps/reference/plot_distance.md):
  exploitation point movement through parameter space

### Vignettes

- `getting-started`: Introduction to APS
- `vector-mode`: Deferred scalarization and rereduce()
- `diagnostics`: Understanding convergence and model quality
