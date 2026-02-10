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
  # Backward compat: old results have exploit_ratio, new have epsilon
  epsilon <- if (!is.null(x$epsilon)) x$epsilon else
    if (!is.null(x$exploit_ratio)) 1 - x$exploit_ratio else 0.5

  # Build data frame
  df <- data.frame(
    performance = x$y,
    iteration = x$iteration
  )

  # Within each iteration, first (1-epsilon) fraction is exploit, rest is explore
  df$action <- "Exploit"
  for (i in 0:n_iter) {
    idx <- which(df$iteration == i)
    if (length(idx) > 0) {
      n_in_iter <- length(idx)
      n_exploit <- n_in_iter - floor(n_in_iter * epsilon)
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


#' Compute Per-Iteration rho* from Out-of-Sample Points
#'
#' For each iteration, computes the maximum achievable correlation (rho*) using
#' all new points evaluated in that iteration (the out-of-sample test set).
#' This matches how cor_out is computed in the main loop.
#'
#' Noise variance (sigma^2) is estimated using nearest neighbors from ALL
#' accumulated data (for density), while signal variance (tau^2) is computed
#' from the out-of-sample points' y variance.
#'
#' @param result An aps_result object
#' @param k Number of nearest neighbors for noise estimation (default 1)
#'
#' @return A data.frame with columns: iteration, rho_star, sigma2, tau2, n_points
#' @keywords internal
compute_per_iteration_rho_star <- function(result, k = 1) {
  y <- result$y
  x_mat <- result$x
  iteration <- result$iteration
  n_iter <- result$n_iter
  n <- length(y)

  # Build nearest neighbor index on all data
  use_fnn <- requireNamespace("FNN", quietly = TRUE)
  if (use_fnn) {
    all_nn <- FNN::get.knn(x_mat, k = k)
  }

  # For each iteration > 0, compute rho* from ALL out-of-sample points
  results <- data.frame(
    iteration = integer(),
    rho_star = numeric(),
    sigma2 = numeric(),
    tau2 = numeric(),
    n_points = integer()
  )

  for (i in 1:n_iter) {
    # Use ALL points from this iteration (both exploit and explore)
    # This matches how cor_out is computed
    iter_idx <- which(iteration == i)

    if (length(iter_idx) < 3) {
      # Not enough points for reliable estimation
      results <- rbind(results, data.frame(
        iteration = i,
        rho_star = NA,
        sigma2 = NA,
        tau2 = NA,
        n_points = length(iter_idx)
      ))
      next
    }

    # Get out-of-sample y values
    y_oos <- y[iter_idx]

    # Estimate sigma^2 using NN from ALL data (for density)
    # For each out-of-sample point, find its NN and compute squared diff
    if (use_fnn) {
      sq_diffs <- numeric(length(iter_idx))
      for (j in seq_along(iter_idx)) {
        orig_idx <- iter_idx[j]
        nn_idx <- all_nn$nn.index[orig_idx, 1:min(k, ncol(all_nn$nn.index))]
        sq_diffs[j] <- mean((y[orig_idx] - y[nn_idx])^2)
      }
      sigma2 <- mean(sq_diffs) / 2
    } else {
      # Fallback: slower O(n^2) computation
      sq_diffs <- numeric(length(iter_idx))
      for (j in seq_along(iter_idx)) {
        orig_idx <- iter_idx[j]
        dists <- rowSums((sweep(x_mat, 2, x_mat[orig_idx, ]))^2)
        dists[orig_idx] <- Inf
        nn_idx <- order(dists)[1:k]
        sq_diffs[j] <- mean((y[orig_idx] - y[nn_idx])^2)
      }
      sigma2 <- mean(sq_diffs) / 2
    }

    # Estimate tau^2 from out-of-sample variance
    var_y_oos <- var(y_oos)
    tau2 <- max(0, var_y_oos - sigma2)

    # Compute rho*
    if (tau2 + sigma2 > 0) {
      rho_star <- sqrt(tau2) / sqrt(tau2 + sigma2)
    } else {
      rho_star <- 0
    }

    results <- rbind(results, data.frame(
      iteration = i,
      rho_star = rho_star,
      sigma2 = sigma2,
      tau2 = tau2,
      n_points = length(iter_idx)
    ))
  }

  results
}


#' Plot Model Correlation Diagnostics
#'
#' Visualizes in-sample and out-of-sample correlations for each model across
#' iterations. Arrows show the change from in-sample to out-of-sample performance.
#' Optionally shows the maximum achievable correlation (rho*) given Monte Carlo
#' noise, computed per-iteration from all out-of-sample points.
#'
#' The green dashed segments show rho* for each iteration block, computed from
#' the same points used to calculate cor_out. rho* may vary across iterations
#' as APS samples different regions with different noise levels.
#'
#' @param x An aps_result object
#' @param competitiveness Threshold for highlighting "good" models (default 0.2)
#' @param show_rho_star Logical. If TRUE, compute and display the maximum
#'   achievable correlation given noise level, per iteration. Default TRUE.
#' @param k Number of nearest neighbors for noise estimation. Default 1.
#' @param ... Additional arguments (ignored)
#'
#' @return A ggplot2 object
#' @export
#'
#' @examples
#' \dontrun{
#' result <- aps(obj_function, n_params = 2)
#' plot_correlation(result)
#' plot_correlation(result, show_rho_star = FALSE)  # without noise adjustment
#' }
plot_correlation <- function(x, competitiveness = 0.2, show_rho_star = TRUE, k = 1, ...) {
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

  # Compute per-iteration rho* if requested
  rho_star_df <- NULL
  subtitle_text <- NULL

  if (show_rho_star && nrow(x$x) >= 10) {
    rho_star_df <- tryCatch(
      compute_per_iteration_rho_star(x, k = k),
      error = function(e) NULL
    )

    if (!is.null(rho_star_df) && any(!is.na(rho_star_df$rho_star))) {
      valid_rho <- rho_star_df$rho_star[!is.na(rho_star_df$rho_star)]
      rho_min <- min(valid_rho)
      rho_max <- max(valid_rho)
      iter_min <- rho_star_df$iteration[which.min(rho_star_df$rho_star)]
      iter_max <- rho_star_df$iteration[which.max(rho_star_df$rho_star)]

      if (abs(rho_max - rho_min) < 0.05) {
        subtitle_text <- sprintf(
          "Green segments = rho* per iteration (%.2f), max achievable given noise",
          mean(valid_rho)
        )
      } else {
        subtitle_text <- sprintf(
          "Green segments = rho* per iteration: %.2f (iter %d) to %.2f (iter %d)",
          rho_min, iter_min, rho_max, iter_max
        )
      }
    }
  }

  # Add model index
  df$model <- 1:nrow(df)

  # Calculate model x-range per iteration for drawing rho* segments
  iter_mins <- stats::aggregate(model ~ iteration, data = df, FUN = min)
  iter_maxs <- stats::aggregate(model ~ iteration, data = df, FUN = max)
  iter_model_ranges <- merge(iter_mins, iter_maxs, by = "iteration")
  names(iter_model_ranges) <- c("iteration", "xmin", "xmax")

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

  # Add per-iteration rho* segments if computed
  if (!is.null(rho_star_df) && any(!is.na(rho_star_df$rho_star))) {
    # Merge rho* with model ranges
    rho_segments <- merge(rho_star_df, iter_model_ranges, by = "iteration")
    rho_segments <- rho_segments[!is.na(rho_segments$rho_star), ]

    if (nrow(rho_segments) > 0) {
      # Add small padding to segments for visual clarity
      rho_segments$xmin <- as.numeric(rho_segments$xmin) - 0.3
      rho_segments$xmax <- as.numeric(rho_segments$xmax) + 0.3

      p <- p +
        ggplot2::geom_segment(
          data = rho_segments,
          ggplot2::aes(x = xmin, xend = xmax, y = rho_star, yend = rho_star),
          linetype = "dashed",
          color = "#2ca02c",
          linewidth = 0.8
        ) +
        ggplot2::labs(subtitle = subtitle_text)
    }
  }

  p
}


#' Plot Exploitation Point Distance Tracking
#'
#' Visualizes cumulative distance traveled by exploitation points across the
#' optimization. Each exploitation point is shown on the x-axis, with the
#' y-axis showing cumulative distance from the starting point.
#'
#' This diagnostic complements rho_tilde by showing WHERE the surrogate thinks
#' the minimum is, not just how well it predicts. Key patterns:
#' \itemize{
#'   \item Steep slope: surrogate moving through parameter space (learning)
#'   \item Flat regions: surrogate staying in same area (converged or stuck)
#'   \item Jumps at iteration boundaries: new models found different region
#'   \item Shaky within iteration: models disagree about minimum location
#'   \item Smooth within iteration: models agree on minimum location
#' }
#'
#' @param x An aps_result object
#' @param normalize Logical. If TRUE, distances are normalized by the parameter
#'   space diameter (max possible distance). Default TRUE.
#' @param ... Additional arguments (ignored)
#'
#' @return A ggplot2 object showing cumulative distance traveled
#' @export
#'
#' @examples
#' \dontrun{
#' result <- aps(obj_function, n_params = 2)
#' plot_distance(result)
#' }
plot_distance <- function(x, normalize = TRUE, ...) {
  if (!inherits(x, "aps_result")) {
    stop("x must be an aps_result object")
  }

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for plotting. Install it with install.packages('ggplot2')")
  }

  # Extract data
  x_mat <- x$x
  iteration <- x$iteration
  n_iter <- x$n_iter
  # Backward compat: old results have exploit_ratio, new have epsilon
  epsilon <- if (!is.null(x$epsilon)) x$epsilon else
    if (!is.null(x$exploit_ratio)) 1 - x$exploit_ratio else 0.5

  # Compute parameter space diameter for normalization
  if (normalize) {
    ranges <- apply(x_mat, 2, function(col) diff(range(col)))
    diameter <- sqrt(sum(ranges^2))
    if (diameter == 0) diameter <- 1
  } else {
    diameter <- 1
  }

  # Extract all exploitation points in order
  exploit_points <- data.frame(
    point_idx = integer(),
    iteration = integer(),
    distance_from_prev = numeric()
  )

  # Also store the x coordinates for distance calculation
  exploit_x_list <- list()

  point_counter <- 0
  for (i in 0:n_iter) {
    iter_idx <- which(iteration == i)
    if (length(iter_idx) == 0) next

    n_in_iter <- length(iter_idx)
    n_exploit <- n_in_iter - floor(n_in_iter * epsilon)
    if (n_exploit == 0) n_exploit <- 1

    # Exploitation points are the first n_exploit in each iteration
    exploit_idx <- iter_idx[1:n_exploit]

    for (j in seq_along(exploit_idx)) {
      point_counter <- point_counter + 1
      current_x <- x_mat[exploit_idx[j], ]

      # Compute distance from previous point
      if (point_counter == 1) {
        dist_from_prev <- 0
      } else {
        prev_x <- exploit_x_list[[point_counter - 1]]
        dist_from_prev <- sqrt(sum((current_x - prev_x)^2)) / diameter
      }

      exploit_x_list[[point_counter]] <- current_x

      exploit_points <- rbind(exploit_points, data.frame(
        point_idx = point_counter,
        iteration = i,
        distance_from_prev = dist_from_prev
      ))
    }
  }

  # Compute cumulative distance
  exploit_points$cumulative_distance <- cumsum(exploit_points$distance_from_prev)

  # Find iteration boundaries for vertical lines
  iter_boundaries <- which(diff(exploit_points$iteration) != 0) + 0.5

  # Build subtitle
  total_dist <- max(exploit_points$cumulative_distance)
  n_points <- nrow(exploit_points)
  if (normalize) {
    subtitle <- sprintf(
      "Total distance: %.2f diameters across %d exploitation points",
      total_dist, n_points
    )
  } else {
    subtitle <- sprintf("Total distance: %.2f across %d points", total_dist, n_points)
  }

  # Create plot
  p <- ggplot2::ggplot(exploit_points, ggplot2::aes(x = point_idx, y = cumulative_distance)) +
    ggplot2::geom_line(color = "#1f78b4", linewidth = 0.8) +
    ggplot2::geom_point(ggplot2::aes(color = factor(iteration)), size = 1.5, alpha = 0.7) +
    ggplot2::geom_vline(xintercept = iter_boundaries, linetype = "dashed",
                        color = "gray50", alpha = 0.5) +
    ggplot2::scale_color_discrete() +
    ggplot2::theme_minimal() +
    ggplot2::theme(legend.position = "none") +
    ggplot2::labs(
      x = "Exploitation Point",
      y = if (normalize) "Cumulative Distance (normalized)" else "Cumulative Distance",
      title = "Exploitation Point Distance Tracking",
      subtitle = subtitle
    )

  p
}


#' Plot Method for aps_result
#'
#' Default plot method showing performance over iterations.
#'
#' @param x An aps_result object
#' @param type Type of plot: "performance" (default), "correlation", or "distance"
#' @param reduction Optional reduction function. If provided, calls rereduce()
#'   before plotting, allowing quick visualization of alternative objectives.
#' @param ... Additional arguments passed to specific plot functions
#'
#' @return A ggplot2 object
#' @export
plot.aps_result <- function(x, type = "performance", reduction = NULL, ...) {
  if (!is.null(reduction)) {
    x <- rereduce(x, reduction)
  }
  switch(type,
    "performance" = plot_performance(x, ...),
    "correlation" = plot_correlation(x, ...),
    "distance" = plot_distance(x, ...),
    stop("Unknown plot type. Use 'performance', 'correlation', or 'distance'")
  )
}
