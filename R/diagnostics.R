#' Plot Performance Statistic Over Iterations
#'
#' Visualizes how the performance statistic evolves across Bayesian optimization
#' iterations, distinguishing between exploration and exploitation points.
#'
#' @param x An aps_result object
#' @param round_digits Number of decimal places for labels
#' @param ... Additional arguments (ignored)
#'
#' @return A ggplot2 object
#' @export
#'
#' @examples
#' \dontrun{
#' result <- aps(obj_function, n_params = 2)
#' plot_performance(result)
#' }
plot_performance <- function(x, round_digits = 3, ...) {
  if (!inherits(x, "aps_result")) {
    stop("x must be an aps_result object")
  }

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for plotting. Install it with install.packages('ggplot2')")
  }

  # Determine exploration vs exploitation
  iter_size <- x$n_obs
  n_iter <- x$n_iter
  exploit_ratio <- if (!is.null(x$exploit_ratio)) x$exploit_ratio else 0.5

  # Build data frame
  df <- data.frame(
    performance = x$y,
    iteration = x$iteration
  )

  # Within each iteration, exploit_ratio fraction is exploit, rest is explore
  df$action <- "Exploit"
  for (i in 0:n_iter) {
    idx <- which(df$iteration == i)
    if (length(idx) > 0) {
      n_in_iter <- length(idx)
      n_exploit <- floor(n_in_iter * exploit_ratio)
      if (n_exploit < n_in_iter) {
        df$action[idx[(n_exploit + 1):n_in_iter]] <- "Explore"
      }
    }
  }

  df$action <- factor(df$action, levels = c("Explore", "Exploit"))

  # Calculate mean performance for exploit points by iteration
  mean_exploit <- stats::aggregate(
    performance ~ iteration,
    data = df[df$action == "Exploit", ],
    FUN = mean
  )
  names(mean_exploit) <- c("iteration", "mean_perf")

  # Create plot
  p <- ggplot2::ggplot(df, ggplot2::aes(x = jitter(iteration), y = performance, color = action)) +
    ggplot2::geom_point(alpha = 0.2) +
    ggplot2::scale_color_manual(
      values = c("Explore" = "#1f78b4", "Exploit" = "#e31a1c"),
      name = "Action Mode"
    ) +
    ggplot2::scale_x_continuous(breaks = 0:n_iter) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      legend.position = "bottom",
      legend.direction = "horizontal"
    ) +
    ggplot2::xlab("Iteration") +
    ggplot2::ylab("Performance Statistic") +
    ggplot2::geom_point(
      data = mean_exploit,
      ggplot2::aes(x = iteration, y = mean_perf),
      inherit.aes = FALSE,
      color = "black",
      size = 2
    ) +
    ggplot2::geom_text(
      data = mean_exploit,
      ggplot2::aes(x = iteration, y = mean_perf, label = round(mean_perf, round_digits)),
      inherit.aes = FALSE,
      vjust = -0.8,
      color = "black",
      size = 3
    )

  p
}


#' Plot Model Correlation Diagnostics
#'
#' Visualizes in-sample and out-of-sample correlations for each model across
#' iterations. Arrows show the change from in-sample to out-of-sample performance.
#'
#' @param x An aps_result object
#' @param competitiveness Threshold for highlighting "good" models (default 0.2)
#' @param ... Additional arguments (ignored)
#'
#' @return A ggplot2 object
#' @export
#'
#' @examples
#' \dontrun{
#' result <- aps(obj_function, n_params = 2)
#' plot_correlation(result)
#' }
plot_correlation <- function(x, competitiveness = 0.2, ...) {
  if (!inherits(x, "aps_result")) {
    stop("x must be an aps_result object")
  }

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for plotting. Install it with install.packages('ggplot2')")
  }

  df <- x$architecture_df

  # Check required columns exist
  if (!all(c("cor_in", "cor_out", "iteration") %in% names(df))) {
    stop("architecture_df missing required columns (cor_in, cor_out, iteration)")
  }

  # Add model index
  df$model <- 1:nrow(df)

  # Calculate which models are "good" within each iteration
  for (i in unique(df$iteration)) {
    idx <- df$iteration == i
    max_in <- max(df$cor_in[idx], na.rm = TRUE)
    max_out <- max(df$cor_out[idx], na.rm = TRUE)

    df$good_model[idx] <- (
      df$cor_in[idx] >= (max_in - competitiveness) &
      df$cor_out[idx] >= (max_out - competitiveness) &
      df$cor_in[idx] > 0 &
      df$cor_out[idx] > 0
    )
  }

  df$alpha <- ifelse(df$good_model, 1, 0.2)

  # Create plot
  p <- ggplot2::ggplot(df) +
    ggplot2::geom_segment(
      ggplot2::aes(x = model, xend = model, y = cor_in, yend = cor_out, alpha = alpha),
      arrow = ggplot2::arrow(length = ggplot2::unit(0.1, "cm")),
      color = "gray30"
    ) +
    ggplot2::geom_point(
      ggplot2::aes(x = model, y = cor_in, alpha = alpha, color = factor(iteration)),
      size = 2
    ) +
    ggplot2::geom_point(
      ggplot2::aes(x = model, y = cor_out, alpha = alpha * 0.8),
      shape = 22,
      fill = "red",
      size = 2
    ) +
    ggplot2::scale_alpha_identity() +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      x = "Model",
      y = "Correlation",
      title = "Model Performance: In-sample (circle) to Out-of-sample (square)"
    ) +
    ggplot2::theme(legend.position = "none")

  p
}


#' Plot Method for aps_result
#'
#' Default plot method showing performance over iterations.
#'
#' @param x An aps_result object
#' @param type Type of plot: "performance" (default) or "correlation"
#' @param ... Additional arguments passed to specific plot functions
#'
#' @return A ggplot2 object
#' @export
plot.aps_result <- function(x, type = "performance", ...) {
  switch(type,
    "performance" = plot_performance(x, ...),
    "correlation" = plot_correlation(x, ...),
    stop("Unknown plot type. Use 'performance' or 'correlation'")
  )
}
