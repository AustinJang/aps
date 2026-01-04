#' Find Minimum via Gradient Descent
#'
#' Uses gradient descent to minimize a function (typically the ensemble prediction
#' or negative uncertainty).
#'
#' @param f_approx A function that takes a torch tensor and returns predictions
#' @param starting_points Matrix of starting points (each row is a point)
#' @param x_min Lower bound for parameters
#' @param x_max Upper bound for parameters
#' @param lr Learning rate
#' @param max_iter Maximum iterations
#' @param tolerance Convergence tolerance
#'
#' @return List with optimized x values and corresponding y values
#' @keywords internal
find_minimum <- function(f_approx,
                         starting_points,
                         x_min,
                         x_max,
                         lr = 0.1,
                         max_iter = 500,
                         tolerance = 0.01) {

  # Convert to tensor with gradient tracking
  x <- torch::torch_tensor(starting_points, requires_grad = TRUE)
  n_obs <- nrow(starting_points)

  # Optimizer
  optimizer <- torch::optim_adam(list(x), lr = lr)

  # Get initial evaluation
  current_eval <- as.array(f_approx(x)$detach())
  count <- 0

  repeat {
    optimizer$zero_grad()

    # Forward pass
    loss <- torch::torch_mean(f_approx(x))

    # Backward pass
    loss$backward()

    # Update
    optimizer$step()

    # Project back to bounds
    torch::with_no_grad({
      x$clamp_(x_min, x_max)
    })

    # Check convergence
    new_eval <- as.array(f_approx(x)$detach())
    max_improvement <- max(current_eval - new_eval)

    count <- count + 1

    if (max_improvement <= tolerance || count >= max_iter) {
      break
    }

    current_eval <- new_eval
  }

  list(
    x = as.matrix(as.array(x$detach())),
    y = as.vector(as.array(f_approx(x)$detach()))
  )
}


#' Propose New Candidate Points
#'
#' Implements the epsilon-LCB acquisition strategy: exploitation
#' (minimize predicted value) and exploration (maximize uncertainty).
#'
#' @param model_list List of trained neural networks
#' @param x_train Current training data
#' @param y_train Current training targets
#' @param x_min Lower bound for parameters
#' @param x_max Upper bound for parameters
#' @param n_candidates Number of candidate points to propose
#' @param exploit_ratio Fraction of candidates for exploitation vs exploration (default 0.5)
#' @param n_starting Number of random starting points for optimization
#' @param tolerance Optimization tolerance
#'
#' @return Matrix of proposed candidate points
#' @keywords internal
propose_candidates <- function(model_list,
                               x_train,
                               y_train,
                               x_min,
                               x_max,
                               n_candidates,
                               exploit_ratio = 0.5,
                               n_starting = 20,
                               tolerance = 0.01) {

  n_params <- ncol(x_train)
  n_models <- length(model_list)
  n_good_models <- sum(sapply(model_list, function(m) !is.null(m)))

  # EXPLOITATION: minimize predictions from each model
  n_exploit <- floor(n_candidates * exploit_ratio)
  exploit_matrix <- NULL

  if (n_good_models >= 1) {
    n_per_model <- ceiling(n_exploit / n_good_models)

    for (i in 1:n_good_models) {
      net <- model_list[[i]]

      # Create prediction function for this model
      f_model <- function(x_tensor) {
        net$eval()
        net(x_tensor)
      }

      # Random starting points
      start_points <- matrix(
        runif(n_candidates * n_starting * n_params, x_min, x_max),
        ncol = n_params
      )

      # Optimize
      result <- find_minimum(
        f_approx = f_model,
        starting_points = start_points,
        x_min = x_min,
        x_max = x_max,
        tolerance = tolerance
      )

      # Take best points
      order_idx <- order(result$y)
      best_points <- result$x[order_idx[1:n_per_model], , drop = FALSE]
      exploit_matrix <- rbind(exploit_matrix, best_points)
    }

    # Trim to exact number needed
    if (nrow(exploit_matrix) > n_exploit) {
      exploit_matrix <- exploit_matrix[1:n_exploit, , drop = FALSE]
    }
  } else {
    # No good models yet - random exploration
    exploit_matrix <- matrix(
      runif(n_exploit * n_params, x_min, x_max),
      ncol = n_params
    )
  }

  # EXPLORATION: maximize uncertainty (minimize negative variance)
  n_explore <- n_candidates - nrow(exploit_matrix)
  explore_matrix <- NULL

  if (n_good_models > 1) {
    # Function to compute negative variance (we minimize this = maximize variance)
    f_uncertainty <- function(x_tensor) {
      preds <- lapply(model_list[1:n_good_models], function(net) {
        net$eval()
        net(x_tensor)
      })

      # Stack predictions and compute variance
      pred_stack <- torch::torch_cat(preds, dim = 2)
      mean_pred <- torch::torch_mean(pred_stack, dim = 2, keepdim = TRUE)
      variance <- torch::torch_mean((pred_stack - mean_pred)^2, dim = 2)

      -variance  # Negative because we minimize
    }

    # Random starting points
    start_points <- matrix(
      runif(n_candidates * n_starting * n_params, x_min, x_max),
      ncol = n_params
    )

    # Optimize
    result <- find_minimum(
      f_approx = f_uncertainty,
      starting_points = start_points,
      x_min = x_min,
      x_max = x_max,
      tolerance = tolerance
    )

    # Take points with highest uncertainty
    order_idx <- order(result$y)  # Most negative = highest variance
    explore_matrix <- result$x[order_idx[1:n_explore], , drop = FALSE]

  } else {
    # Not enough models for variance - random exploration
    explore_matrix <- matrix(
      runif(n_explore * n_params, x_min, x_max),
      ncol = n_params
    )
  }

  # Combine exploitation and exploration points
  rbind(exploit_matrix, explore_matrix)
}
