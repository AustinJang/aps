#' @keywords internal
"_PACKAGE"

#' @importFrom stats cor rnorm runif sd var
#' @importFrom utils flush.console
NULL

# Silence R CMD check notes for ggplot2 aes variables
utils::globalVariables(c(
  "iteration", "performance", "action", "mean_perf",
  "model", "cor_in", "cor_out", "alpha", "xmin", "xmax", "rho_star",
  "point_idx", "cumulative_distance"
))
