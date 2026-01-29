#' Adversarial Parameter Selection
#'
#' Main function for adversarial parameter selection using Bayesian optimization
#' with deep ensembles. Iteratively trains an ensemble of neural networks to
#' approximate the objective function, then uses the ensemble to select parameter
#' values likely to produce poor method performance.
#'
#' @param obj_function Function that takes a parameter vector and returns a scalar
#'   performance statistic. Lower values indicate worse performance (adversarial).
#' @param n_params Number of parameters in the search space
#' @param x_min Lower bound for all parameters (scalar or vector)
#' @param x_max Upper bound for all parameters (scalar or vector)
#' @param n_iter Number of Bayesian optimization iterations
#' @param n_obs Number of new observations per iteration
#' @param n_models Number of models in the ensemble
#' @param x_init Optional matrix of initial parameter values
#' @param y_init Optional vector of initial performance statistics
#' @param num_cores Number of cores for parallel evaluation (default 1)
#' @param exploit_ratio Fraction of candidates for exploitation vs exploration (default 0.5)
#' @param competitiveness_param Threshold for eliminating underperforming models (default 0.2)
#' @param tolerance Optimization convergence tolerance (default 0.01)
#' @param verbose Print progress messages (default TRUE)
#' @param checkpoint_file Optional file path to save checkpoints after each iteration.
#'   Allows resuming if the process is interrupted.
#' @param resume_from Optional. Either a file path to a checkpoint or an aps_result
#'   object to resume from. When resuming, iterations continue from where they left off.
#' @param use_genetic Logical. If TRUE, use genetic algorithm-style mutations to
#'   generate new architectures from well-performing parents. If FALSE (default),
#'   use purely random architecture generation.
#'
#' @return A list with class "aps_result" containing:
#'   \item{x}{Matrix of all evaluated parameter values}
#'   \item{y}{Vector of all performance statistics}
#'   \item{architecture_df}{Data frame of model architectures and performance}
#'   \item{best_x}{Parameter values with lowest (worst) performance}
#'   \item{best_y}{Lowest performance statistic found}
#'   \item{iteration}{Vector indicating which iteration each point was from}
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Simple example: find parameters where sin function is minimized
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
#' print(result$best_x)
#' print(result$best_y)
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
                num_cores = 1,
                exploit_ratio = 0.5,
                competitiveness_param = 0.2,
                tolerance = 0.01,
                verbose = TRUE,
                checkpoint_file = NULL,
                resume_from = NULL,
                use_genetic = FALSE) {

  # Handle bounds
  if (length(x_min) == 1) x_min <- rep(x_min, n_params)
  if (length(x_max) == 1) x_max <- rep(x_max, n_params)

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
    y_train <- checkpoint$y
    iteration_tracker <- checkpoint$iteration
    architecture_history <- checkpoint$architecture_df
    start_iter <- max(iteration_tracker) + 1

    # Restore current architecture state (the "keep" models from last iteration)
    last_iter_arch <- architecture_history[architecture_history$iteration == max(architecture_history$iteration), ]
    architecture_df <- last_iter_arch[last_iter_arch$keep, c("depth", "width", "reg", "dropout", "activation")]
    n_bad_models <- n_models - nrow(architecture_df)

    # Fill in with new architectures if needed
    if (n_bad_models > 0) {
      if (use_genetic && nrow(architecture_df) > 0) {
        architecture_df <- rbind(architecture_df, mutate_architectures(architecture_df, n_bad_models))
      } else {
        architecture_df <- rbind(architecture_df, generate_architectures(n_bad_models))
      }
    }

    if (verbose) {
      message(sprintf("Resumed from iteration %d", start_iter - 1))
      message(sprintf("Existing data: %d points", nrow(x_train)))
      message(sprintf("Continuing to iteration %d", n_iter))
    }

    # Check if already done
    if (start_iter > n_iter) {
      if (verbose) message("Already completed requested iterations. Returning existing result.")
      return(checkpoint)
    }

  } else {
    # Generate initial data if not provided
    if (is.null(x_init) || is.null(y_init)) {
      if (verbose) message("Generating initial random sample...")

      x_train <- matrix(nrow = n_obs, ncol = n_params)
      for (j in 1:n_params) {
        x_train[, j] <- runif(n_obs, x_min[j], x_max[j])
      }

      y_train <- evaluate_objective(obj_function, x_train, num_cores)
      iteration_tracker <- rep(0, n_obs)  # Iteration 0 = initialization
    } else {
      x_train <- x_init
      y_train <- y_init
      iteration_tracker <- rep(0, nrow(x_init))
    }

    # Initialize architecture
    architecture_df <- generate_architectures(n_models)
    n_bad_models <- n_models  # All models are "new" initially
  }

  # Main optimization loop
  for (i in start_iter:n_iter) {
    n_good_models <- n_models - n_bad_models

    if (verbose) {
      message(sprintf("\n========== ITERATION %d ==========", i))
      message(sprintf("Training data: %d points", nrow(x_train)))
      message(sprintf("Good models from previous: %d", n_good_models))
    }

    # Train ensemble
    if (verbose) message("Training ensemble...")
    model_list <- train_ensemble(
      x_train, y_train,
      architecture_df,
      num_cores = num_cores,
      verbose = verbose
    )

    # Propose new candidates
    if (verbose) message("Proposing candidate points...")
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
    if (verbose) message("Evaluating candidates...")
    new_y <- evaluate_objective(obj_function, new_x, num_cores)

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
          mutate_architectures(new_architecture_df, n_bad_models)
        )
      } else {
        # Pure random generation
        new_architecture_df <- rbind(
          new_architecture_df,
          generate_architectures(n_bad_models)
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
        architecture_df = architecture_history,
        best_x = x_train[best_idx_tmp, ],
        best_y = y_train[best_idx_tmp],
        iteration = iteration_tracker,
        n_iter = n_iter,
        n_obs = n_obs,
        exploit_ratio = exploit_ratio
      )
      class(checkpoint_result) <- "aps_result"
      saveRDS(checkpoint_result, file = checkpoint_file)
      if (verbose) message(sprintf("Checkpoint saved to %s", checkpoint_file))
    }
  }

  # Find best (most adversarial) point
  best_idx <- which.min(y_train)

  result <- list(
    x = x_train,
    y = y_train,
    architecture_df = architecture_history,
    best_x = x_train[best_idx, ],
    best_y = y_train[best_idx],
    iteration = iteration_tracker,
    n_iter = n_iter,
    n_obs = n_obs,
    exploit_ratio = exploit_ratio
  )

  class(result) <- "aps_result"
  result
}


#' Evaluate Objective Function
#'
#' Helper function to evaluate the objective function, optionally in parallel.
#'
#' @param obj_function The objective function
#' @param x Matrix of parameter values
#' @param num_cores Number of cores
#'
#' @return Vector of performance statistics
#' @keywords internal
evaluate_objective <- function(obj_function, x, num_cores = 1) {
  if (num_cores == 1) {
    apply(x, 1, obj_function)
  } else {
    parallel::mclapply(
      1:nrow(x),
      function(i) obj_function(x[i, ]),
      mc.cores = num_cores
    ) |> unlist()
  }
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

  cat("Performance Statistics:\n")
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

  invisible(object)
}
