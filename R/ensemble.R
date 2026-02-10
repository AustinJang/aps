#' Create a Neural Network Module
#'
#' Creates a feedforward neural network with specified architecture.
#'
#' @param input_dim Number of input features (parameters)
#' @param width Number of units in hidden layers
#' @param depth Number of hidden layers
#' @param activation Activation function ("relu", "sigmoid", or "tanh")
#' @param dropout Dropout rate (0 to 1)
#'
#' @return A torch nn_module
#' @keywords internal
create_network <- function(input_dim, width, depth, activation = "relu", dropout = 0.0) {
  # Select activation function

  act_fn <- switch(activation,
    "relu" = torch::nn_relu,
    "sigmoid" = torch::nn_sigmoid,
    "tanh" = torch::nn_tanh,
    torch::nn_relu  # default

)

  # Build layer list
  layers <- list()

  # First layer
  layers[[1]] <- torch::nn_linear(input_dim, width)
  layers[[2]] <- act_fn()

  # Hidden layers
  if (depth > 1) {
    for (i in 2:depth) {
      layers[[length(layers) + 1]] <- torch::nn_linear(width, width)
      layers[[length(layers) + 1]] <- act_fn()
      if (dropout > 0) {
        layers[[length(layers) + 1]] <- torch::nn_dropout(p = dropout)
      }
    }
  }

  # Output layer
  layers[[length(layers) + 1]] <- torch::nn_linear(width, 1)

  # Create sequential model
  do.call(torch::nn_sequential, layers)
}


#' Train a Single Neural Network
#'
#' Trains a neural network on the provided data with early stopping.
#'
#' @param net A torch nn_module
#' @param x_train Training features (matrix)
#' @param y_train Training targets (vector)
#' @param epochs Maximum number of epochs
#' @param batch_size Batch size for training
#' @param lr Learning rate
#' @param patience Early stopping patience
#' @param validation_split Fraction of data for validation
#' @param verbose Print progress
#'
#' @return Trained network (modified in place)
#' @keywords internal
train_network <- function(net,
                          x_train,
                          y_train,
                          epochs = 1000,
                          batch_size = 32,
                          lr = 0.001,
                          patience = 20,
                          validation_split = 0.2,
                          verbose = FALSE) {
  # Convert to tensors
  n <- nrow(x_train)
  n_val <- floor(n * validation_split)
  n_train <- n - n_val

  # Shuffle and split
  idx <- sample(n)
  train_idx <- idx[1:n_train]
  val_idx <- idx[(n_train + 1):n]

  x_train_t <- torch::torch_tensor(x_train[train_idx, , drop = FALSE])
  y_train_t <- torch::torch_tensor(matrix(y_train[train_idx], ncol = 1))
  x_val_t <- torch::torch_tensor(x_train[val_idx, , drop = FALSE])
  y_val_t <- torch::torch_tensor(matrix(y_train[val_idx], ncol = 1))

  # Optimizer
  optimizer <- torch::optim_adam(net$parameters, lr = lr)

  # Early stopping state
  best_val_loss <- Inf
  patience_counter <- 0
  best_state <- NULL

  # Training loop
for (epoch in 1:epochs) {
    net$train()

    # Mini-batch training
    perm <- sample(n_train)
    total_loss <- 0
    n_batches <- ceiling(n_train / batch_size)

    for (b in 1:n_batches) {
      start_idx <- (b - 1) * batch_size + 1
      end_idx <- min(b * batch_size, n_train)
      batch_idx <- perm[start_idx:end_idx]

      optimizer$zero_grad()
      pred <- net(x_train_t[batch_idx, , drop = FALSE])
      loss <- torch::nnf_mse_loss(pred, y_train_t[batch_idx, , drop = FALSE])
      loss$backward()
      optimizer$step()

      total_loss <- total_loss + loss$item()
    }

    # Validation
    net$eval()
    with_no_grad <- torch::with_no_grad
    val_loss <- with_no_grad({
      val_pred <- net(x_val_t)
      torch::nnf_mse_loss(val_pred, y_val_t)$item()
    })

    # Early stopping check
    if (val_loss < best_val_loss - 1e-4) {
      best_val_loss <- val_loss
      patience_counter <- 0
      best_state <- lapply(net$parameters, function(p) p$clone())
    } else {
      patience_counter <- patience_counter + 1
    }

    if (patience_counter >= patience) {
      if (verbose) {
        message(sprintf("Early stopping at epoch %d (val_loss: %.4f)", epoch, best_val_loss))
      }
      break
    }

    if (verbose && epoch %% 100 == 0) {
      message(sprintf("Epoch %d - train_loss: %.4f, val_loss: %.4f",
                      epoch, total_loss / n_batches, val_loss))
    }
  }

  # Restore best weights
  if (!is.null(best_state)) {
    for (i in seq_along(net$parameters)) {
      net$parameters[[i]]$set_data(best_state[[i]])
    }
  }

  invisible(net)
}


# Safe sample that handles length-1 vectors correctly.
# R's sample(x, n) treats a single integer x as sample(1:x, n), which is wrong
# when we want to sample from a vector that happens to have one element.
safe_sample <- function(x, size, replace = FALSE) {
  if (length(x) == 1L) rep(x, size) else sample(x, size, replace = replace)
}


#' Get Tier Bounds
#'
#' Maps a tier integer to the maximum allowed depth and width. Tier 0 uses the
#' smallest values from the config; each subsequent tier unlocks the next step.
#'
#' @param tier Integer tier level (0-based)
#' @param config Optional architecture config list (with \code{depth} and \code{width})
#'
#' @return List with \code{max_depth}, \code{max_width}, \code{min_depth}, \code{min_width}
#' @keywords internal
get_tier_bounds <- function(tier, config = NULL) {
  depth_steps <- sort(unique(
    if (!is.null(config) && !is.null(config$depth)) config$depth else 1:5
  ))
  width_steps <- sort(unique(
    if (!is.null(config) && !is.null(config$width)) config$width else c(32, 64, 128, 256, 512)
  ))

  d_idx <- min(tier + 1, length(depth_steps))
  w_idx <- min(tier + 1, length(width_steps))

  list(
    max_depth = depth_steps[d_idx],
    max_width = width_steps[w_idx],
    min_depth = depth_steps[1],
    min_width = width_steps[1]
  )
}


#' Compute Maximum Tier
#'
#' Derives the maximum tier from the architecture config's depth and width ranges.
#' The max tier is the number of steps needed to unlock all depth/width values.
#'
#' @param config Optional architecture config list
#'
#' @return Integer maximum tier
#' @keywords internal
compute_max_tier <- function(config = NULL) {
  n_depth <- length(unique(
    if (!is.null(config) && !is.null(config$depth)) config$depth else 1:5
  ))
  n_width <- length(unique(
    if (!is.null(config) && !is.null(config$width)) config$width else c(32, 64, 128, 256, 512)
  ))
  max(n_depth, n_width) - 1
}


#' Next Width Step
#'
#' Given a current width, returns the next larger width in the config's width
#' sequence. If already at max, returns current width.
#'
#' @param current_width Current width value
#' @param config Optional architecture config list
#'
#' @return Integer next width step
#' @keywords internal
next_width_step <- function(current_width, config = NULL) {
  width_options <- sort(unique(
    if (!is.null(config) && !is.null(config$width)) config$width else c(32, 64, 128, 256, 512)
  ))
  above <- width_options[width_options > current_width]
  if (length(above) > 0) above[1] else current_width
}


#' Infer Tier from Existing Architectures
#'
#' Determines the minimum tier that accommodates the architectures already in use.
#' Used when resuming from a checkpoint that doesn't store the tier.
#'
#' @param arch_df Data frame of current architectures
#' @param config Optional architecture config list
#'
#' @return Integer tier
#' @keywords internal
infer_tier <- function(arch_df, config = NULL) {
  max_d <- max(arch_df$depth)
  max_w <- max(arch_df$width)
  max_possible <- compute_max_tier(config)

  for (t in 0:max_possible) {
    bounds <- get_tier_bounds(t, config)
    if (bounds$max_depth >= max_d && bounds$max_width >= max_w) {
      return(t)
    }
  }
  max_possible
}


#' Generate Random Architecture
#'
#' Generates a random neural network architecture specification with uniform
#' width across all hidden layers. This is a reasonable default for most use cases.
#'
#' For custom architectures (e.g., funnel/bottleneck shapes, per-layer activations),
#' users can either: (1) provide their own architecture_df to aps(), or
#' (2) modify this function to generate different architecture specifications.
#' The required columns are: depth, width, reg, dropout, activation.
#'
#' @param n Number of architectures to generate
#' @param config Optional list specifying architecture parameter ranges. If NULL,
#'   uses defaults. Supported keys:
#'   \itemize{
#'     \item \code{depth}: Integer vector of allowed depths (default: 1:5)
#'     \item \code{width}: Integer vector of allowed widths (default: c(32, 64, 128, 256, 512))
#'     \item \code{reg}: Numeric vector of allowed regularization values (default: seq(0, 0.0005, 0.0001))
#'     \item \code{dropout}: Numeric vector of allowed dropout values (default: seq(0, 0.5, 0.1))
#'     \item \code{activation}: Character vector of allowed activations (default: c("relu", "sigmoid", "tanh"))
#'   }
#' @param tier Optional integer tier level. When provided, depth and width are
#'   capped at the tier's bounds (see \code{get_tier_bounds}). If NULL (default),
#'   full config ranges are used (backward compatible).
#'
#' @return A data frame with architecture specifications
#' @export
generate_architectures <- function(n, config = NULL, tier = NULL) {
  # Default configuration
  defaults <- list(
    depth = 1:5,
    width = c(32, 64, 128, 256, 512),
    reg = seq(0, 0.0005, by = 0.0001),
    dropout = seq(0, 0.5, by = 0.1),
    activation = c("relu", "sigmoid", "tanh")
  )

  # Merge user config with defaults
  if (!is.null(config)) {
    for (key in names(config)) {
      if (key %in% names(defaults)) {
        defaults[[key]] <- config[[key]]
      } else {
        warning(sprintf("Unknown architecture_config key '%s' ignored", key))
      }
    }
  }

  # Apply tier constraints to depth and width
  if (!is.null(tier)) {
    bounds <- get_tier_bounds(tier, config)
    defaults$depth <- defaults$depth[defaults$depth <= bounds$max_depth]
    defaults$width <- defaults$width[defaults$width <= bounds$max_width]
    if (length(defaults$depth) == 0) defaults$depth <- bounds$max_depth
    if (length(defaults$width) == 0) defaults$width <- bounds$max_width
  }

  data.frame(
    depth = safe_sample(defaults$depth, n, replace = TRUE),
    width = safe_sample(defaults$width, n, replace = TRUE),
    reg = safe_sample(defaults$reg, n, replace = TRUE),
    dropout = safe_sample(defaults$dropout, n, replace = TRUE),
    activation = safe_sample(defaults$activation, n, replace = TRUE),
    stringsAsFactors = FALSE
  )
}


#' Mutate Architectures (Genetic Algorithm)
#'
#' Creates mutations of parent architectures by randomly perturbing parameters.
#' Used for genetic algorithm-style architecture search where good-performing
#' architectures are mutated to create the next generation.
#'
#' Mutations include:
#' \itemize{
#'   \item depth: +1 or -1 (bounded by config)
#'   \item width: multiply by 2 or divide by 2 (bounded by config)
#'   \item activation: randomly swap to a different function from config
#'   \item reg/dropout: small random perturbation (bounded by config)
#' }
#'
#' @param parents Data frame of parent architectures to mutate
#' @param n Number of mutated children to generate
#' @param mutation_rate Probability of mutating each parameter (default 0.3)
#' @param config Optional list specifying architecture parameter ranges (same format
#'   as \code{generate_architectures}). Used to bound mutations.
#' @param tier Optional integer tier level. When provided, depth and width mutations
#'   are capped at the tier's bounds, and the incremental constraint is enforced:
#'   depth and width cannot both increase relative to the parent in a single mutation.
#'
#' @return A data frame with mutated architecture specifications
#' @export
mutate_architectures <- function(parents, n, mutation_rate = 0.3, config = NULL, tier = NULL) {
  if (nrow(parents) == 0) {
    return(generate_architectures(n, config, tier = tier))
  }

  # Default configuration (same as generate_architectures)
  defaults <- list(
    depth = 1:5,
    width = c(32, 64, 128, 256, 512),
    reg = seq(0, 0.0005, by = 0.0001),
    dropout = seq(0, 0.5, by = 0.1),
    activation = c("relu", "sigmoid", "tanh")
  )

  # Merge user config with defaults
  if (!is.null(config)) {
    for (key in names(config)) {
      if (key %in% names(defaults)) {
        defaults[[key]] <- config[[key]]
      }
    }
  }

  # Extract bounds from config, applying tier cap if active
  depth_min <- min(defaults$depth)
  width_min <- min(defaults$width)
  if (!is.null(tier)) {
    bounds <- get_tier_bounds(tier, config)
    depth_max <- bounds$max_depth
    width_max <- bounds$max_width
  } else {
    depth_max <- max(defaults$depth)
    width_max <- max(defaults$width)
  }
  reg_max <- max(defaults$reg)
  dropout_max <- max(defaults$dropout)
  activations <- defaults$activation

  children <- data.frame(
    depth = integer(n),
    width = integer(n),
    reg = numeric(n),
    dropout = numeric(n),
    activation = character(n),
    stringsAsFactors = FALSE
  )

  for (i in 1:n) {
    # Select random parent
    parent <- parents[sample(nrow(parents), 1), ]

    # Mutate depth
    if (runif(1) < mutation_rate) {
      delta <- sample(c(-1, 1), 1)
      new_depth <- parent$depth + delta
      children$depth[i] <- max(depth_min, min(depth_max, new_depth))
    } else {
      children$depth[i] <- parent$depth
    }

    # Mutate width
    if (runif(1) < mutation_rate) {
      factor <- sample(c(0.5, 2), 1)
      new_width <- round(parent$width * factor)
      children$width[i] <- max(width_min, min(width_max, new_width))
    } else {
      children$width[i] <- parent$width
    }

    # Incremental constraint: can't increase both depth and width vs parent
    if (!is.null(tier)) {
      if (children$depth[i] > parent$depth && children$width[i] > parent$width) {
        # Roll back one dimension (randomly)
        if (runif(1) < 0.5) {
          children$depth[i] <- parent$depth
        } else {
          children$width[i] <- parent$width
        }
      }
    }

    # Mutate activation (cheap dimension, no constraint)
    if (runif(1) < mutation_rate) {
      other_acts <- setdiff(activations, parent$activation)
      if (length(other_acts) > 0) {
        children$activation[i] <- sample(other_acts, 1)
      } else {
        children$activation[i] <- parent$activation
      }
    } else {
      children$activation[i] <- parent$activation
    }

    # Mutate regularization (cheap dimension)
    if (runif(1) < mutation_rate) {
      children$reg[i] <- max(0, min(reg_max, parent$reg + rnorm(1, 0, 0.0001)))
    } else {
      children$reg[i] <- parent$reg
    }

    # Mutate dropout (cheap dimension)
    if (runif(1) < mutation_rate) {
      children$dropout[i] <- max(0, min(dropout_max, parent$dropout + rnorm(1, 0, 0.1)))
    } else {
      children$dropout[i] <- parent$dropout
    }
  }

  children
}


#' Train a Single Model (Helper for Parallelization)
#'
#' Creates and trains a single neural network. Used internally for parallel training.
#'
#' @param arch_row Single row from architecture data frame
#' @param x_train Training features
#' @param y_train Training targets
#' @param n_params Number of input parameters
#' @param ... Additional arguments passed to train_network
#'
#' @return Trained network
#' @keywords internal
train_single_model <- function(arch_row, x_train, y_train, n_params, ...) {
  # Create network
net <- create_network(
    input_dim = n_params,
    width = arch_row$width,
    depth = arch_row$depth,
    activation = arch_row$activation,
    dropout = arch_row$dropout
  )

  # Train network
  train_network(net, x_train, y_train, verbose = FALSE, ...)

  net
}


#' Train Deep Ensemble
#'
#' Trains an ensemble of neural networks with varying architectures.
#' Supports parallel training across multiple CPU cores.
#'
#' @param x_train Training features (matrix)
#' @param y_train Training targets (vector)
#' @param architecture_df Data frame specifying architectures
#' @param num_cores Number of CPU cores for parallel training (default 1 = sequential)
#' @param verbose Print progress
#' @param ... Additional arguments passed to train_network
#'
#' @return List of trained networks
#' @keywords internal
train_ensemble <- function(x_train, y_train, architecture_df, num_cores = 1, verbose = TRUE, ...) {
  n_models <- nrow(architecture_df)
  n_params <- ncol(x_train)

  # Note: Parallel model training with torch and fork() doesn't work reliably on macOS

  # due to Metal/MPS threading issues. The num_cores parameter is kept for potential
  # future use on Linux systems where fork() works better with torch.
  # For now, model training is always sequential.

  # Sequential training
  model_list <- vector("list", n_models)

  for (i in 1:n_models) {
    arch <- architecture_df[i, ]

    model_list[[i]] <- train_single_model(arch, x_train, y_train, n_params, ...)

    if (verbose) {
      message(sprintf("Model %d/%d trained (depth=%d, width=%d, act=%s)",
                      i, n_models, arch$depth, arch$width, arch$activation))
    }
  }

  model_list
}


#' Ensemble Prediction
#'
#' Get predictions from all models in the ensemble.
#'
#' @param model_list List of trained networks
#' @param x Input features (matrix)
#'
#' @return List with mean prediction and variance (uncertainty)
#' @export
ensemble_predict <- function(model_list, x) {
  x_t <- torch::torch_tensor(x)

  # Get predictions from each model
  preds <- lapply(model_list, function(net) {
    net$eval()
    torch::with_no_grad({
      as.array(net(x_t))
    })
  })

  # Stack predictions
  pred_matrix <- do.call(cbind, preds)

  list(
    mean = rowMeans(pred_matrix),
    variance = apply(pred_matrix, 1, var),
    predictions = pred_matrix
  )
}
