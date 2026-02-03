#' Adversarial Parameter Selection
#'
#' Main function for adversarial parameter selection using Bayesian optimization
#' with deep ensembles. Iteratively trains an ensemble of neural networks to
#' approximate the objective function, then uses the ensemble to select parameter
#' values likely to produce poor method performance.
#'
#' @param obj_function Function that takes a parameter vector and returns either:
#'   \itemize{
#'     \item A scalar performance statistic (classic usage), or
#'     \item A named numeric vector of intermediate outputs (vector mode).
#'   }
#'   In vector mode, a \code{reduction} function must be provided to map the
#'   output vector to a scalar. Lower values indicate worse performance.
#' @param n_params Number of parameters in the search space
#' @param x_min Lower bound for all parameters (scalar or vector)
#' @param x_max Upper bound for all parameters (scalar or vector)
#' @param n_iter Number of Bayesian optimization iterations
#' @param n_obs Number of new observations per iteration
#' @param n_models Number of models in the ensemble
#' @param x_init Optional matrix of initial parameter values
#' @param y_init Optional vector of initial reduced performance statistics.
#'   For vector mode, you can also pass \code{y_raw_init} instead.
#' @param y_raw_init Optional matrix of initial raw objective outputs (vector
#'   mode only). One row per observation, one column per output component.
#'   If provided with a \code{reduction}, \code{y_init} is computed automatically.
#' @param num_cores Number of cores for parallel evaluation (default 1)
#' @param exploit_ratio Fraction of candidates for exploitation vs exploration (default 0.5)
#' @param competitiveness_param Threshold for eliminating underperforming models (default 0.2)
#' @param tolerance Optimization convergence tolerance (default 0.01)
#' @param verbose Print progress messages (default TRUE)
#' @param checkpoint_file Optional file path to save checkpoints after each iteration.
#'   Allows resuming if the process is interrupted.
#' @param resume_from Optional. Either a file path to a checkpoint or an aps_result
#'   object to resume from. When resuming, iterations continue from where they left off.
#'   If the \code{reduction} function has changed since the previous run, the
#'   stored raw outputs (\code{y_raw}) are re-scalarized without re-running
#'   simulations. The surrogate is then retrained on the new scalar targets.
#' @param use_genetic Logical. If TRUE, use genetic algorithm-style mutations to
#'   generate new architectures from well-performing parents. If FALSE (default),
#'   use purely random architecture generation.
#' @param reduction Optional function that maps a named numeric vector (one row
#'   of objective output) to a scalar. Required when \code{obj_function} returns
#'   vectors. Can be changed on resume to re-scalarize existing data without
#'   re-running simulations.
#'   Examples: \code{function(v) -max(v)}, \code{function(v) v["bias_A"] - v["bias_B"]}.
#'   If NULL and \code{obj_function} returns a scalar, behaves identically to the
#'   original package (full backward compatibility).
#' @param output_names Optional character vector naming the outputs of
#'   \code{obj_function}. If NULL, names are inferred from the first evaluation.
#'   Only relevant in vector mode.
#' @param architecture_config Optional list specifying neural network architecture
#'   parameter ranges for the ensemble. If NULL, uses defaults. Supported keys:
#'   \itemize{
#'     \item \code{depth}: Integer vector of allowed depths (default: 1:5)
#'     \item \code{width}: Integer vector of allowed widths (default: c(32, 64, 128, 256, 512))
#'     \item \code{reg}: Numeric vector of allowed L2 regularization values (default: seq(0, 0.0005, 0.0001))
#'     \item \code{dropout}: Numeric vector of allowed dropout values (default: seq(0, 0.5, 0.1))
#'     \item \code{activation}: Character vector of allowed activations (default: c("relu", "sigmoid", "tanh"))
#'   }
#'   Example: \code{architecture_config = list(depth = 1:3, activation = c("relu", "tanh"))}
#' @param early_stop Logical. If TRUE, stop early when the surrogate has learned
#'   all learnable signal (rho_tilde >= threshold) AND the minimum has been stable
#'   for \code{min_stable_iters} iterations. Default FALSE.
#' @param rho_tilde_threshold Threshold for the reliability ratio (rho_out / rho*)
#'   above which the surrogate is considered to have learned all signal. Default 0.95.
#' @param min_stable_iters Number of consecutive iterations without improvement in
#'   the minimum before early stopping is triggered. Default 2.
#'
#' @return A list with class "aps_result" containing:
#'   \item{x}{Matrix of all evaluated parameter values}
#'   \item{y}{Vector of all (reduced) performance statistics}
#'   \item{y_raw}{Matrix of raw objective outputs (NULL in scalar mode).
#'     Rows = evaluations, columns = output components.}
#'   \item{architecture_df}{Data frame of model architectures and performance}
#'   \item{best_x}{Parameter values with lowest (worst) performance}
#'   \item{best_y}{Lowest performance statistic found}
#'   \item{iteration}{Vector indicating which iteration each point was from}
#'   \item{reduction}{The reduction function used (NULL in scalar mode)}
#'   \item{output_names}{Names of the output vector components (NULL in scalar mode)}
#'   \item{stopped_early}{Logical indicating if early stopping was triggered}
#'   \item{final_iteration}{The last iteration completed (may be less than n_iter if stopped early)}
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Scalar mode (classic, fully backward-compatible):
#' obj_fn <- function(x) sin(x[1]) * cos(x[2])
#'
#' result <- aps(
#'   obj_function = obj_fn,
#'   n_params = 2,
#'   x_min = 0,
#'   x_max = 2 * pi,
#'   n_iter = 5,
#'   n_obs = 100
#' )
#'
#' # Vector mode: objective returns multiple metrics, reduce to scalar
#' obj_fn_vec <- function(x) c(bias_A = x[1]^2, bias_B = (x[2]-1)^2)
#'
#' result <- aps(
#'   obj_function = obj_fn_vec,
#'   n_params = 2,
#'   x_min = -2,
#'   x_max = 2,
#'   n_iter = 5,
#'   n_obs = 100,
#'   reduction = function(v) v["bias_A"] - v["bias_B"]
#' )
#'
#' # Resume with a different reduction — no re-simulation needed:
#' result2 <- aps(
#'   obj_function = obj_fn_vec,
#'   n_params = 2,
#'   x_min = -2,
#'   x_max = 2,
#'   n_iter = 10,
#'   n_obs = 100,
#'   reduction = function(v) -v["bias_B"],
#'   resume_from = result
#' )
#' }
aps <- function(obj_function,
                n_params,
                x_min = 0,
                x_max = 1,
                n_iter = 10,
                n_obs = 500,
                n_models = 5,
                x_init = NULL,
                y_init = NULL,
                y_raw_init = NULL,
                num_cores = 1,
                exploit_ratio = 0.5,
                competitiveness_param = 0.2,
                tolerance = 0.01,
                verbose = TRUE,
                checkpoint_file = NULL,
                resume_from = NULL,
                use_genetic = FALSE,
                reduction = NULL,
                output_names = NULL,
                architecture_config = NULL,
                early_stop = FALSE,
                rho_tilde_threshold = 0.95,
                min_stable_iters = 2) {

  # Handle bounds
  if (length(x_min) == 1) x_min <- rep(x_min, n_params)
  if (length(x_max) == 1) x_max <- rep(x_max, n_params)

  # Determine mode: scalar vs vector
  # vector_mode is set to TRUE once we confirm obj_function returns vectors
  vector_mode <- !is.null(reduction)
  y_raw_train <- NULL  # Will be a matrix in vector mode, NULL in scalar mode

  # Initialize storage
  architecture_history <- NULL
  iteration_tracker <- NULL
  start_iter <- 1


  # Handle resume from checkpoint
  if (!is.null(resume_from)) {
    # Load checkpoint if it's a file path
    if (is.character(resume_from)) {
      if (verbose) message(sprintf("Loading checkpoint from %s...", resume_from))
      checkpoint <- readRDS(resume_from)
    } else if (inherits(resume_from, "aps_result")) {
      checkpoint <- resume_from
    } else {
      stop("resume_from must be a file path or an aps_result object")
    }

    # Restore state
    x_train <- checkpoint$x
    iteration_tracker <- checkpoint$iteration
    architecture_history <- checkpoint$architecture_df
    start_iter <- max(iteration_tracker) + 1

    # Restore seeds if available
    seeds_train <- checkpoint$seeds

    # Restore raw outputs if available
    if (!is.null(checkpoint$y_raw)) {
      y_raw_train <- checkpoint$y_raw
      vector_mode <- TRUE

      # Infer output_names from stored data
      if (is.null(output_names) && !is.null(colnames(y_raw_train))) {
        output_names <- colnames(y_raw_train)
      }
    }

    # Restore or recompute reduced y
    if (vector_mode && !is.null(y_raw_train)) {
      if (!is.null(reduction)) {
        # Re-scalarize with (possibly new) reduction function
        if (verbose) message("Applying reduction function to stored raw outputs...")
        y_train <- apply(y_raw_train, 1, reduction)
      } else {
        # No reduction provided on resume — use stored y as-is
        y_train <- checkpoint$y
      }
    } else {
      y_train <- checkpoint$y
    }

    # Restore current architecture state (the "keep" models from last iteration)
    last_iter_arch <- architecture_history[architecture_history$iteration == max(architecture_history$iteration), ]
    architecture_df <- last_iter_arch[last_iter_arch$keep, c("depth", "width", "reg", "dropout", "activation")]
    n_bad_models <- n_models - nrow(architecture_df)

    # Fill in with new architectures if needed
    if (n_bad_models > 0) {
      if (use_genetic && nrow(architecture_df) > 0) {
        architecture_df <- rbind(architecture_df, mutate_architectures(architecture_df, n_bad_models, config = architecture_config))
      } else {
        architecture_df <- rbind(architecture_df, generate_architectures(n_bad_models, config = architecture_config))
      }
    }

    if (verbose) {
      message(sprintf("Resumed from iteration %d", start_iter - 1))
      message(sprintf("Existing data: %d points", nrow(x_train)))
      if (vector_mode) message(sprintf("Vector mode: %d output components", ncol(y_raw_train)))
      message(sprintf("Continuing to iteration %d", n_iter))
    }

    # Check if already done
    if (start_iter > n_iter) {
      if (verbose) message("Already completed requested iterations. Returning existing result.")
      return(checkpoint)
    }

  } else {
    # Generate initial data if not provided
    if (is.null(x_init) || (is.null(y_init) && is.null(y_raw_init))) {
      if (verbose) message("Generating initial random sample...")

      x_train <- matrix(nrow = n_obs, ncol = n_params)
      for (j in 1:n_params) {
        x_train[, j] <- runif(n_obs, x_min[j], x_max[j])
      }

      eval_result <- evaluate_objective(obj_function, x_train, num_cores,
                                        reduction = reduction,
                                        output_names = output_names)
      y_train <- eval_result$y
      y_raw_train <- eval_result$y_raw
      seeds_train <- eval_result$seeds
      output_names <- eval_result$output_names
      if (!is.null(y_raw_train)) vector_mode <- TRUE

      iteration_tracker <- rep(0, n_obs)  # Iteration 0 = initialization

    } else if (!is.null(y_raw_init)) {
      # User provided raw initial outputs
      x_train <- x_init
      y_raw_train <- y_raw_init
      seeds_train <- rep(NA_integer_, nrow(x_init))
      vector_mode <- TRUE
      if (is.null(output_names) && !is.null(colnames(y_raw_init))) {
        output_names <- colnames(y_raw_init)
      }
      if (!is.null(reduction)) {
        y_train <- apply(y_raw_train, 1, reduction)
      } else {
        stop("reduction is required when y_raw_init is provided")
      }
      iteration_tracker <- rep(0, nrow(x_init))

    } else {
      x_train <- x_init
      y_train <- y_init
      seeds_train <- rep(NA_integer_, nrow(x_init))
      iteration_tracker <- rep(0, nrow(x_init))
    }

    # Initialize architecture
    architecture_df <- generate_architectures(n_models, config = architecture_config)
    n_bad_models <- n_models  # All models are "new" initially
  }

  # Early stopping tracking
  prev_best_y <- min(y_train)
  stable_iter_count <- 0
  stopped_early <- FALSE

  # Main optimization loop
  for (i in start_iter:n_iter) {
    n_good_models <- n_models - n_bad_models

    if (verbose) {
      message(sprintf("\n========== ITERATION %d/%d ==========", i, n_iter))
      message(sprintf("Training data: %d points | Good models: %d",
                      nrow(x_train), n_good_models))
      cat(sprintf("\r[%d/%d] Training ensemble...", i, n_iter))
    }

    # Train ensemble (always on reduced y)
    model_list <- train_ensemble(
      x_train, y_train,
      architecture_df,
      num_cores = num_cores,
      verbose = verbose
    )

    # Propose new candidates
    if (verbose) cat(sprintf("\r[%d/%d] Proposing candidates...    ", i, n_iter))
    new_x <- propose_candidates(
      model_list = model_list,
      x_train = x_train,
      y_train = y_train,
      x_min = x_min,
      x_max = x_max,
      n_candidates = n_obs,
      exploit_ratio = exploit_ratio,
      tolerance = tolerance
    )

    # Evaluate new candidates
    if (verbose) cat(sprintf("\r[%d/%d] Evaluating %d candidates...", i, n_iter, n_obs))
    eval_result <- evaluate_objective(obj_function, new_x, num_cores,
                                      reduction = reduction,
                                      output_names = output_names)
    new_y <- eval_result$y
    new_y_raw <- eval_result$y_raw
    new_seeds <- eval_result$seeds

    # Update architecture performance tracking
    architecture_df$cor_in <- NA
    architecture_df$cor_out <- NA

    for (j in 1:n_models) {
      pred_old <- ensemble_predict(list(model_list[[j]]), x_train)$mean
      pred_new <- ensemble_predict(list(model_list[[j]]), new_x)$mean

      architecture_df$cor_in[j] <- cor(pred_old, y_train)
      architecture_df$cor_out[j] <- cor(pred_new, new_y)
    }

    # Add data
    x_train <- rbind(x_train, new_x)
    y_train <- c(y_train, new_y)
    seeds_train <- c(seeds_train, new_seeds)
    if (vector_mode && !is.null(new_y_raw)) {
      y_raw_train <- rbind(y_raw_train, new_y_raw)
    }
    iteration_tracker <- c(iteration_tracker, rep(i, nrow(new_x)))

    # Determine which architectures to keep
    architecture_df[is.na(architecture_df)] <- -1
    best_cor_in <- max(architecture_df$cor_in)
    best_cor_out <- max(architecture_df$cor_out)

    architecture_df$keep <- (
      architecture_df$cor_in > best_cor_in - competitiveness_param &
      architecture_df$cor_in > 0 &
      architecture_df$cor_out > best_cor_out - competitiveness_param &
      architecture_df$cor_out > 0
    )

    # Record architecture history
    architecture_df$iteration <- i
    architecture_history <- rbind(architecture_history, architecture_df)

    # Generate new architectures to replace bad ones
    n_bad_models <- sum(!architecture_df$keep)
    if (verbose) {
      cat("\r", strrep(" ", 60), "\r", sep = "")  # Clear progress line
      message(sprintf("Models kept: %d, replaced: %d", n_models - n_bad_models, n_bad_models))
      message(sprintf("Best in-sample cor: %.3f, Best out-of-sample cor: %.3f",
                      best_cor_in, best_cor_out))
      message(sprintf("Current minimum y: %.4f", min(y_train)))
    }

    # Update architecture for next iteration
    new_architecture_df <- architecture_df[architecture_df$keep,
                                           c("depth", "width", "reg", "dropout", "activation")]

    if (n_bad_models > 0) {
      if (use_genetic && nrow(new_architecture_df) > 0) {
        # Mutate from good parents
        new_architecture_df <- rbind(
          new_architecture_df,
          mutate_architectures(new_architecture_df, n_bad_models, config = architecture_config)
        )
      } else {
        # Pure random generation
        new_architecture_df <- rbind(
          new_architecture_df,
          generate_architectures(n_bad_models, config = architecture_config)
        )
      }
    }
    architecture_df <- new_architecture_df

    # Save checkpoint if requested
    if (!is.null(checkpoint_file)) {
      best_idx_tmp <- which.min(y_train)
      checkpoint_result <- list(
        x = x_train,
        y = y_train,
        y_raw = y_raw_train,
        seeds = seeds_train,
        architecture_df = architecture_history,
        best_x = x_train[best_idx_tmp, ],
        best_y = y_train[best_idx_tmp],
        iteration = iteration_tracker,
        n_iter = n_iter,
        n_obs = n_obs,
        exploit_ratio = exploit_ratio,
        reduction = reduction,
        output_names = output_names
      )
      class(checkpoint_result) <- "aps_result"
      saveRDS(checkpoint_result, file = checkpoint_file)
      if (verbose) message(sprintf("Checkpoint saved to %s", checkpoint_file))
    }

    # Early stopping check
    if (early_stop && i >= 2) {
      # Check if minimum improved
      current_best_y <- min(y_train)
      improved <- current_best_y < prev_best_y - 1e-6

      if (improved) {
        stable_iter_count <- 0
        prev_best_y <- current_best_y
      } else {
        stable_iter_count <- stable_iter_count + 1
      }

      # Compute rho_tilde for this iteration
      # Estimate sigma^2 and tau^2 from the new points
      if (requireNamespace("FNN", quietly = TRUE) && length(new_y) >= 3) {
        nn <- FNN::get.knn(new_x, k = 1)
        sq_diffs <- (new_y - new_y[nn$nn.index[, 1]])^2
        sigma2_iter <- mean(sq_diffs) / 2
        var_y_iter <- var(new_y)
        tau2_iter <- max(0, var_y_iter - sigma2_iter)

        if (tau2_iter + sigma2_iter > 0) {
          rho_star_iter <- sqrt(tau2_iter) / sqrt(tau2_iter + sigma2_iter)
          rho_tilde_iter <- if (rho_star_iter > 0) best_cor_out / rho_star_iter else NA
          rho_tilde_iter <- min(1, rho_tilde_iter)  # Cap at 1
        } else {
          rho_tilde_iter <- NA
        }
      } else {
        rho_tilde_iter <- NA
      }

      # Check stopping condition
      if (!is.na(rho_tilde_iter) &&
          rho_tilde_iter >= rho_tilde_threshold &&
          stable_iter_count >= min_stable_iters) {
        if (verbose) {
          message(sprintf("\n*** EARLY STOP at iteration %d ***", i))
          message(sprintf("rho_tilde = %.3f (>= %.3f threshold)", rho_tilde_iter, rho_tilde_threshold))
          message(sprintf("Minimum stable for %d iterations", stable_iter_count))
        }
        stopped_early <- TRUE
        break
      }

      if (verbose && !is.na(rho_tilde_iter)) {
        message(sprintf("Early stop check: rho_tilde=%.3f, stable_iters=%d",
                        rho_tilde_iter, stable_iter_count))
      }
    }
  }

  # Find best (most adversarial) point
  best_idx <- which.min(y_train)

  result <- list(
    x = x_train,
    y = y_train,
    y_raw = y_raw_train,
    seeds = seeds_train,
    architecture_df = architecture_history,
    best_x = x_train[best_idx, ],
    best_y = y_train[best_idx],
    iteration = iteration_tracker,
    n_iter = n_iter,
    n_obs = n_obs,
    exploit_ratio = exploit_ratio,
    reduction = reduction,
    output_names = output_names,
    stopped_early = stopped_early,
    final_iteration = max(iteration_tracker)
  )

  class(result) <- "aps_result"
  result
}


#' Evaluate Objective Function
#'
#' Helper function to evaluate the objective function, optionally in parallel.
#' Handles both scalar and vector-valued objective functions.
#'
#' @param obj_function The objective function
#' @param x Matrix of parameter values
#' @param num_cores Number of cores
#' @param reduction Optional reduction function for vector outputs
#' @param output_names Optional names for output components
#'
#' @return List with:
#'   \item{y}{Vector of (reduced) scalar performance statistics}
#'   \item{y_raw}{Matrix of raw outputs (NULL if obj_function returns scalars)}
#'   \item{output_names}{Names of output components (NULL if scalar)}
#' @keywords internal
evaluate_objective <- function(obj_function, x, num_cores = 1,
                               reduction = NULL, output_names = NULL) {
  n <- nrow(x)

  # Generate one seed per evaluation for reproducibility
  seeds <- sample.int(.Machine$integer.max, n)

  # Evaluate all points, setting seed before each call
  if (num_cores == 1) {
    raw_results <- lapply(1:n, function(i) {
      set.seed(seeds[i])
      obj_function(x[i, ])
    })
  } else {
    raw_results <- parallel::mclapply(
      1:n,
      function(i) {
        set.seed(seeds[i])
        obj_function(x[i, ])
      },
      mc.cores = num_cores
    )
  }

  # Detect scalar vs vector mode from first result
  first <- raw_results[[1]]
  is_vector <- length(first) > 1

  if (!is_vector) {
    # Scalar mode — classic behavior
    y <- vapply(raw_results, function(r) as.numeric(r), numeric(1))
    return(list(y = y, y_raw = NULL, output_names = NULL, seeds = seeds))
  }

  # Vector mode
  if (is.null(reduction)) {
    stop("obj_function returned a vector of length ", length(first),
         " but no reduction function was provided. ",
         "Supply a reduction function to map the output vector to a scalar.")
  }

  # Infer output names
  if (is.null(output_names) && !is.null(names(first))) {
    output_names <- names(first)
  }
  n_outputs <- length(first)

  # Build raw output matrix
  y_raw <- matrix(nrow = n, ncol = n_outputs)
  if (!is.null(output_names)) colnames(y_raw) <- output_names

  for (i in seq_along(raw_results)) {
    y_raw[i, ] <- as.numeric(raw_results[[i]])
  }

  # Apply reduction
  y <- apply(y_raw, 1, reduction)

  list(y = y, y_raw = y_raw, output_names = output_names, seeds = seeds)
}


#' Re-reduce an APS result with a new reduction function
#'
#' Given an existing aps_result with stored raw outputs (\code{y_raw}),
#' recompute the scalar \code{y} values using a new reduction function.
#' This allows exploring different scalarizations without re-running
#' expensive simulations.
#'
#' @param result An aps_result object with non-NULL \code{y_raw}
#' @param reduction A function mapping a named numeric vector to a scalar
#'
#' @return A modified aps_result with updated \code{y}, \code{best_x},
#'   \code{best_y}, and \code{reduction}
#'
#' @export
rereduce <- function(result, reduction) {
  if (!inherits(result, "aps_result")) {
    stop("result must be an aps_result object")
  }
  if (is.null(result$y_raw)) {
    stop("Cannot rereduce: result has no stored raw outputs (y_raw). ",
         "The original run used scalar mode.")
  }

  result$y <- apply(result$y_raw, 1, reduction)
  result$reduction <- reduction

  best_idx <- which.min(result$y)
  result$best_x <- result$x[best_idx, ]
  result$best_y <- result$y[best_idx]

  result
}


#' Compute noise-adjusted convergence diagnostic (reliability ratio)
#'
#' Computes the reliability ratio, which adjusts the raw out-of-sample
#' correlation for irreducible noise. A perfect surrogate achieves
#' \eqn{\rho^* = \tau / \sqrt{\tau^2 + \sigma^2}}, not 1, due to Monte Carlo
#' noise. The reliability ratio \eqn{\tilde{\rho} = \rho_{out} / \rho^*}
#' indicates how close the surrogate is to this theoretical maximum.
#'
#' Noise variance \eqn{\sigma^2} is estimated using nearest-neighbor differences
#' (Gasser et al. 1986), which does not require repeated evaluations at the
#' same point.
#'
#' @param result An aps_result object
#' @param k Number of nearest neighbors to use for variance estimation (default 1).
#'   Larger k reduces variance but increases bias.
#'
#' @return A list with class "aps_reliability" containing:
#'   \item{sigma2}{Estimated noise variance (average MC noise)}
#'   \item{tau2}{Estimated signal variance}
#'   \item{rho_star}{Maximum achievable correlation given noise level}
#'   \item{rho_out}{Best out-of-sample correlation from the final iteration}
#'   \item{rho_tilde}{Reliability ratio: rho_out / rho_star}
#'   \item{snr}{Signal-to-noise ratio: tau2 / sigma2}
#'   \item{interpretation}{A text interpretation of the diagnostic}
#'
#' @details
#' The diagnostic answers the question: "Is my surrogate as good as it can be
#' given the noise level?" If \eqn{\tilde{\rho} \approx 1}, the surrogate has
#' extracted all learnable signal. If \eqn{\tilde{\rho} \ll 1}, there is room
#' for improvement (more data, better architecture, etc.).
#'
#' The noise variance is estimated using the nearest-neighbor difference

#' estimator: for each point, we find its nearest neighbor in parameter space
#' and compute the squared difference in y values. Under smoothness assumptions,
#' this differences out the signal, leaving only noise.
#'
#' @references
#' Gasser, T., Sroka, L., & Jennen-Steinmetz, C. (1986). Residual variance and
#' residual pattern in nonlinear regression. Biometrika, 73(3), 625-633.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' result <- aps(obj_function = my_sim, n_params = 5, ...)
#' diag <- reliability_ratio(result)
#' print(diag)
#' }
reliability_ratio <- function(result, k = 1) {
  if (!inherits(result, "aps_result")) {
    stop("result must be an aps_result object")
  }

  x <- result$x
  y <- result$y
  n <- length(y)

  if (n < 10) {
    stop("Need at least 10 observations for reliable variance estimation")
  }

  # --- Estimate sigma^2 using nearest-neighbor differences ---

  # Check if FNN is available for fast nearest neighbor search
  if (requireNamespace("FNN", quietly = TRUE)) {
    # Use FNN for efficient KD-tree nearest neighbor search
    nn <- FNN::get.knn(x, k = k)
    nn_idx <- nn$nn.index  # n x k matrix of neighbor indices

    # Compute squared differences for each point and its k neighbors
    sq_diffs <- numeric(n * k)
    for (i in 1:n) {
      for (j in 1:k) {
        neighbor_idx <- nn_idx[i, j]
        sq_diffs[(i - 1) * k + j] <- (y[i] - y[neighbor_idx])^2
      }
    }

    # sigma^2 estimate: E[(y_i - y_nn)^2] / 2
    sigma2 <- mean(sq_diffs) / 2

  } else {
    # Fallback: naive O(n^2) nearest neighbor search
    message("FNN package not found, using slower O(n^2) nearest neighbor search. ",
            "Install FNN for faster computation: install.packages('FNN')")

    # Compute pairwise distances
    sq_diffs <- numeric(n)
    for (i in 1:n) {
      # Find k nearest neighbors
      dists <- rowSums((sweep(x, 2, x[i, ]))^2)
      dists[i] <- Inf  # Exclude self
      nn_idx <- order(dists)[1:k]

      # Average squared difference to k neighbors
      sq_diffs[i] <- mean((y[i] - y[nn_idx])^2)
    }

    sigma2 <- mean(sq_diffs) / 2
  }

  # --- Estimate tau^2 = Var(y) - sigma^2 ---
  var_y <- var(y)
  tau2 <- max(0, var_y - sigma2)  # Truncate at 0

  # --- Compute rho_star = tau / sqrt(tau^2 + sigma^2) ---
  if (tau2 + sigma2 > 0) {
    rho_star <- sqrt(tau2) / sqrt(tau2 + sigma2)
  } else {
    rho_star <- 0
  }

 # --- Get rho_out from architecture history ---
  arch_df <- result$architecture_df

  if (is.null(arch_df) || nrow(arch_df) == 0) {
    warning("No architecture history found. Cannot compute rho_out.")
    rho_out <- NA
    rho_tilde <- NA
  } else {
    # Get the final iteration's results
    last_iter <- max(arch_df$iteration)
    last_arch <- arch_df[arch_df$iteration == last_iter, ]

    # Use the best out-of-sample correlation among models
    rho_out <- max(last_arch$cor_out, na.rm = TRUE)

    # Compute reliability ratio
    if (rho_star > 0) {
      rho_tilde <- rho_out / rho_star
      # Cap at 1 (can exceed due to estimation noise)
      rho_tilde <- min(1, rho_tilde)
    } else {
      rho_tilde <- NA
    }
  }

  # --- Signal-to-noise ratio ---
  snr <- if (sigma2 > 0) tau2 / sigma2 else Inf

  # --- Interpretation ---
  if (tau2 == 0 && sigma2 > 0) {
    interpretation <- sprintf(
      "Warning: Estimated noise variance (%.3f) exceeds total variance (%.3f). ",
      sigma2, var_y
    )
    interpretation <- paste0(interpretation,
      "This suggests the response surface may be dominated by noise with little ",
      "learnable signal. The surrogate cannot meaningfully predict outcomes. ",
      "Consider: (1) increasing replications per evaluation to reduce MC noise, or ",
      "(2) the parameter space may not strongly affect the outcome.")
  } else if (is.na(rho_tilde)) {
    interpretation <- "Could not compute reliability ratio (missing data or zero variance)."
  } else if (rho_tilde >= 0.9) {
    interpretation <- sprintf(
      "Excellent (rho_tilde = %.2f). The surrogate has extracted nearly all learnable signal. ",
      rho_tilde
    )
    interpretation <- paste0(interpretation,
      "Continued optimization is unlikely to find substantially better regions.")
  } else if (rho_tilde >= 0.7) {
    interpretation <- sprintf(
      "Good (rho_tilde = %.2f). The surrogate is performing well but may improve with more data.",
      rho_tilde
    )
  } else if (rho_tilde >= 0.5) {
    interpretation <- sprintf(
      "Moderate (rho_tilde = %.2f). The surrogate has room for improvement. ",
      rho_tilde
    )
    interpretation <- paste0(interpretation,
      "Consider running more iterations or adjusting architecture diversity.")
  } else {
    interpretation <- sprintf(
      "Poor (rho_tilde = %.2f). The surrogate is underperforming. ",
      rho_tilde
    )
    interpretation <- paste0(interpretation,
      "The response surface may be difficult to learn, or more data is needed.")
  }

  # Add context about noise level
  if (!is.na(snr)) {
    if (snr < 1) {
      interpretation <- paste0(interpretation,
        sprintf("\n\nNote: Low signal-to-noise ratio (SNR = %.2f). ", snr),
        "The simulation noise dominates the signal variation.")
    } else if (snr > 10) {
      interpretation <- paste0(interpretation,
        sprintf("\n\nNote: High signal-to-noise ratio (SNR = %.2f). ", snr),
        "The response surface has strong signal relative to noise.")
    }
  }

  result_obj <- list(
    sigma2 = sigma2,
    tau2 = tau2,
    rho_star = rho_star,
    rho_out = rho_out,
    rho_tilde = rho_tilde,
    snr = snr,
    var_y = var_y,
    n = n,
    k = k,
    interpretation = interpretation
  )

  class(result_obj) <- "aps_reliability"
  result_obj
}


#' Print reliability ratio diagnostic
#'
#' @param x An aps_reliability object
#' @param ... Additional arguments (ignored)
#'
#' @export
print.aps_reliability <- function(x, ...) {
  cat("APS Convergence Diagnostic: Reliability Ratio\n")
  cat("==============================================\n\n")

  cat(sprintf("Sample size: %d points\n", x$n))
  cat(sprintf("Nearest neighbors used: k = %d\n\n", x$k))

  cat("Variance Decomposition:\n")
  cat(sprintf("  Total variance (Var(y)):  %.4f\n", x$var_y))
  cat(sprintf("  Noise variance (sigma^2): %.4f\n", x$sigma2))
  cat(sprintf("  Signal variance (tau^2):  %.4f\n", x$tau2))
  cat(sprintf("  Signal-to-noise ratio:    %.2f\n\n", x$snr))

  cat("Correlation Diagnostic:\n")
  cat(sprintf("  Maximum achievable (rho*):   %.3f\n", x$rho_star))
  cat(sprintf("  Observed out-of-sample:      %.3f\n", x$rho_out))
  cat(sprintf("  Reliability ratio (rho~):    %.3f\n\n", x$rho_tilde))

  cat("Interpretation:\n")
  cat(strwrap(x$interpretation, width = 70, prefix = "  "), sep = "\n")
  cat("\n")

  invisible(x)
}


#' Print APS Result
#'
#' @param x An aps_result object
#' @param ... Additional arguments (ignored)
#'
#' @export
print.aps_result <- function(x, ...) {
  cat("Adversarial Parameter Selection Result\n")
  cat("======================================\n")
  cat(sprintf("Iterations: %d\n", x$n_iter))
  cat(sprintf("Total evaluations: %d\n", length(x$y)))
  if (!is.null(x$y_raw)) {
    cat(sprintf("Output mode: vector (%d components)\n", ncol(x$y_raw)))
    if (!is.null(x$output_names)) {
      cat(sprintf("Output names: %s\n", paste(x$output_names, collapse = ", ")))
    }
  }
  cat(sprintf("Best (minimum) y: %.4f\n", x$best_y))
  cat(sprintf("Best parameters: %s\n", paste(round(x$best_x, 4), collapse = ", ")))
  invisible(x)
}


#' Summary of APS Result
#'
#' @param object An aps_result object
#' @param ... Additional arguments (ignored)
#'
#' @export
summary.aps_result <- function(object, ...) {
  cat("Adversarial Parameter Selection Summary\n")
  cat("=======================================\n\n")

  if (!is.null(object$y_raw)) {
    cat(sprintf("Output mode: vector (%d components)\n", ncol(object$y_raw)))
    if (!is.null(object$output_names)) {
      cat(sprintf("Output names: %s\n", paste(object$output_names, collapse = ", ")))
    }
    cat("\nRaw Output Statistics (across all evaluations):\n")
    for (j in 1:ncol(object$y_raw)) {
      nm <- if (!is.null(object$output_names)) object$output_names[j] else sprintf("output_%d", j)
      cat(sprintf("  %s: min=%.4f, mean=%.4f, max=%.4f\n",
                  nm, min(object$y_raw[, j], na.rm = TRUE),
                  mean(object$y_raw[, j], na.rm = TRUE),
                  max(object$y_raw[, j], na.rm = TRUE)))
    }
    cat("\n")
  }

  cat("Reduced Performance Statistics:\n")
  cat(sprintf("  Min (adversarial): %.4f\n", min(object$y)))
  cat(sprintf("  Max: %.4f\n", max(object$y)))
  cat(sprintf("  Mean: %.4f\n", mean(object$y)))
  cat(sprintf("  SD: %.4f\n", sd(object$y)))

  cat("\nBy Iteration:\n")
  for (i in 0:object$n_iter) {
    idx <- object$iteration == i
    if (sum(idx) > 0) {
      cat(sprintf("  Iter %d: n=%d, min=%.4f, mean=%.4f\n",
                  i, sum(idx), min(object$y[idx]), mean(object$y[idx])))
    }
  }

  cat("\nBest Adversarial Point:\n")
  cat(sprintf("  y = %.4f\n", object$best_y))
  cat(sprintf("  x = [%s]\n", paste(round(object$best_x, 4), collapse = ", ")))

  if (!is.null(object$y_raw)) {
    best_idx <- which.min(object$y)
    cat("  Raw outputs at best point:\n")
    raw_best <- object$y_raw[best_idx, ]
    for (j in seq_along(raw_best)) {
      nm <- if (!is.null(object$output_names)) object$output_names[j] else sprintf("output_%d", j)
      cat(sprintf("    %s = %.4f\n", nm, raw_best[j]))
    }
  }

  invisible(object)
}