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
#'
#' @return A data frame with architecture specifications
#' @export
generate_architectures <- function(n) {
  data.frame(
    depth = sample(1:5, n, replace = TRUE),
    width = 2^(4 + sample(1:5, n, replace = TRUE)),  # 32, 64, 128, 256, 512
    reg = sample(0:5, n, replace = TRUE) * 0.0001,
    dropout = sample(0:5, n, replace = TRUE) * 0.1,
    activation = sample(c("relu", "sigmoid", "tanh"), n, replace = TRUE),
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
#'   \item depth: +1 or -1 (bounded to 1-5)
#'   \item width: multiply by 2 or divide by 2 (bounded to 32-512)
#'   \item activation: randomly swap to a different function
#'   \item reg/dropout: small random perturbation
#' }
#'
#' @param parents Data frame of parent architectures to mutate
#' @param n Number of mutated children to generate
#' @param mutation_rate Probability of mutating each parameter (default 0.3)
#'
#' @return A data frame with mutated architecture specifications
#' @export
mutate_architectures <- function(parents, n, mutation_rate = 0.3) {
  if (nrow(parents) == 0) {
    return(generate_architectures(n))
  }

  activations <- c("relu", "sigmoid", "tanh")
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
      children$depth[i] <- max(1, min(5, parent$depth + delta))
    } else {
      children$depth[i] <- parent$depth
    }

    # Mutate width
    if (runif(1) < mutation_rate) {
      factor <- sample(c(0.5, 2), 1)
      children$width[i] <- max(32, min(512, round(parent$width * factor)))
    } else {
      children$width[i] <- parent$width
    }

    # Mutate activation
    if (runif(1) < mutation_rate) {
      other_acts <- setdiff(activations, parent$activation)
      children$activation[i] <- sample(other_acts, 1)
    } else {
      children$activation[i] <- parent$activation
    }

    # Mutate regularization
    if (runif(1) < mutation_rate) {
      children$reg[i] <- max(0, parent$reg + rnorm(1, 0, 0.0001))
    } else {
      children$reg[i] <- parent$reg
    }

    # Mutate dropout
    if (runif(1) < mutation_rate) {
      children$dropout[i] <- max(0, min(0.5, parent$dropout + rnorm(1, 0, 0.1)))
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
