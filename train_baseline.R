# Training script: sg-LASSO-MIDAS baseline.
#
# Trains the sg-LASSO-MIDAS baseline (single-threaded sparsegl) on the
# full panel. The neural variants are trained by the companion script
# train_nam_family.R; both scripts share the same data split through
# set.seed(s * iter), so the baseline and neural worker files can be
# joined afterwards into a single summary table.
#
# Default: 10 iterations, horizons t = c(8, 8.5, 9), 30 tasks in total.
# Reads:   period_final_us_nonfin.csv
# Writes:  baseline_worker_s<>_t<>_iter<>.rds
#
# Each iteration is dispatched to its own worker; the iterations are
# independent and parallelise cleanly.

rm(list = ls())

source('import functions for empirical application.R')

library(sparsegl)
library(midasml)
library(dplyr)
library(RSpectra)
library(pROC)
library(openxlsx)
library(ggplot2)
library(xtable)
library(caret)
library(survival)
library(parallel)
library(foreach)
library(doParallel)
library(PRROC)
library(MLmetrics)
library(survivalROC)
library(pracma)
library(dotCall64)
library(rlang)
library(readxl)
library(Survivalml)
library(lubridate)
library(timeROC)
library(torch)

cat("torch check:", as.character(torch_tensor(1)$item()), "\n")

single_agg_net <- nn_module(
  "SingleAggNet",
  initialize = function(d_in, L, hidden_size = 16L, n_layers = 1L) {
    self$n_layers <- as.integer(n_layers)
    self$act <- nn_elu()
    self$fc1 <- nn_linear(d_in, hidden_size)
    floor_w <- as.integer(max(4L, as.integer(L)))
    if (self$n_layers >= 2L) {
      h2 <- as.integer(max(floor(hidden_size / 2), floor_w))
      self$fc2 <- nn_linear(hidden_size, h2)
      if (self$n_layers >= 3L) {
        h3 <- as.integer(max(floor(h2 / 2), floor_w))
        self$fc3 <- nn_linear(h2, h3)
        if (self$n_layers >= 4L) {
          h4 <- as.integer(max(floor(h3 / 2), floor_w))
          self$fc4 <- nn_linear(h3, h4)
          self$fc_out <- nn_linear(h4, L)
        } else {
          self$fc_out <- nn_linear(h3, L)
        }
      } else {
        self$fc_out <- nn_linear(h2, L)
      }
    } else {
      self$fc_out <- nn_linear(hidden_size, L)
    }
  },
  forward = function(x) {
    x <- self$act(self$fc1(x))
    if (self$n_layers >= 2L) {
      x <- self$act(self$fc2(x))
      if (self$n_layers >= 3L) {
        x <- self$act(self$fc3(x))
        if (self$n_layers >= 4L) x <- self$act(self$fc4(x))
      }
    }
    self$fc_out(x)
  }
)

single_pred_net <- nn_module(
  "SinglePredNet",
  initialize = function(L_in, hidden_size = 8L, n_layers = 1L, p_dropout = 0.2) {
    self$n_layers <- as.integer(n_layers)
    self$act <- nn_relu()
    self$drop <- nn_dropout(p = p_dropout)
    self$fc1 <- nn_linear(L_in, hidden_size)
    if (self$n_layers >= 2L) {
      h2 <- as.integer(max(floor(hidden_size / 2), 4L))
      self$fc2 <- nn_linear(hidden_size, h2)
      if (self$n_layers >= 3L) {
        h3 <- as.integer(max(floor(h2 / 2), 4L))
        self$fc3 <- nn_linear(h2, h3)
        if (self$n_layers >= 4L) {
          h4 <- as.integer(max(floor(h3 / 2), 4L))
          self$fc4 <- nn_linear(h3, h4)
          self$fc_out <- nn_linear(h4, 1L)
        } else {
          self$fc_out <- nn_linear(h3, 1L)
        }
      } else {
        self$fc_out <- nn_linear(h2, 1L)
      }
    } else {
      self$fc_out <- nn_linear(hidden_size, 1L)
    }
  },
  forward = function(x) {
    x <- self$drop(self$act(self$fc1(x)))
    if (self$n_layers >= 2L) {
      x <- self$drop(self$act(self$fc2(x)))
      if (self$n_layers >= 3L) {
        x <- self$drop(self$act(self$fc3(x)))
        if (self$n_layers >= 4L) x <- self$drop(self$act(self$fc4(x)))
      }
    }
    self$fc_out(x)
  }
)

build_joint_pred <- function(input_dim, H_pred, n_pred_layers, p_dropout) {
  input_dim <- as.integer(input_dim)
  H_pred <- as.integer(H_pred)
  if (n_pred_layers == 1L) {
    nn_sequential(
      nn_linear(input_dim, H_pred), nn_relu(), nn_dropout(p = p_dropout),
      nn_linear(H_pred, 1L))
  } else if (n_pred_layers == 2L) {
    H2 <- as.integer(max(floor(H_pred / 2), 8L))
    nn_sequential(
      nn_linear(input_dim, H_pred), nn_relu(), nn_dropout(p = p_dropout),
      nn_linear(H_pred, H2), nn_relu(), nn_dropout(p = p_dropout),
      nn_linear(H2, 1L))
  } else if (n_pred_layers == 3L) {
    H2 <- as.integer(max(floor(H_pred / 2), 8L))
    H3 <- as.integer(max(floor(H_pred / 4), 4L))
    nn_sequential(
      nn_linear(input_dim, H_pred), nn_relu(), nn_dropout(p = p_dropout),
      nn_linear(H_pred, H2), nn_relu(), nn_dropout(p = p_dropout),
      nn_linear(H2, H3), nn_relu(), nn_dropout(p = p_dropout),
      nn_linear(H3, 1L))
  } else {
    H2 <- as.integer(max(floor(H_pred / 2), 8L))
    H3 <- as.integer(max(floor(H_pred / 4), 4L))
    H4 <- as.integer(max(floor(H_pred / 8), 4L))
    nn_sequential(
      nn_linear(input_dim, H_pred), nn_relu(), nn_dropout(p = p_dropout),
      nn_linear(H_pred, H2), nn_relu(), nn_dropout(p = p_dropout),
      nn_linear(H2, H3), nn_relu(), nn_dropout(p = p_dropout),
      nn_linear(H3, H4), nn_relu(), nn_dropout(p = p_dropout),
      nn_linear(H4, 1L))
  }
}

deep_midas_module <- nn_module(
  "DeepMIDAS",
  initialize = function(d, K, L = 3L, H_agg = 16L, H_pred = 32L,
                        n_pred_layers = 1L, n_agg_layers = 1L,
                        p_dropout = 0.2, p_group_dropout = 0.1,
                        use_neural_agg = TRUE) {
    self$K <- as.integer(K)
    self$L <- as.integer(L)
    self$p_group_dropout <- p_group_dropout
    self$use_neural_agg <- use_neural_agg

    if (use_neural_agg) {
      self$agg_net <- single_agg_net(as.integer(d), self$L, as.integer(H_agg),
                                     n_layers = as.integer(n_agg_layers))
    }

    input_dim <- as.integer(self$K * self$L + 1L)
    self$pred <- build_joint_pred(input_dim, H_pred, n_pred_layers, p_dropout)
  },
  forward = function(Z_lags, W_fixed = NULL) {
    batch_size <- Z_lags[[1]]$shape[1]
    h_list <- vector("list", self$K)
    for (k in seq_len(self$K)) {
      if (self$use_neural_agg) {
        h_list[[k]] <- self$agg_net(Z_lags[[k]])
      } else {
        h_list[[k]] <- torch_mm(Z_lags[[k]], W_fixed)
      }
    }
    h <- torch_stack(h_list, dim = 2)
    if (self$training && self$p_group_dropout > 0) {
      mask <- (torch_rand(batch_size, self$K, 1L) > self$p_group_dropout)
      mask <- mask$to(dtype = h$dtype, device = h$device)
      h <- h * mask
    }
    h_flat <- h$reshape(c(batch_size, -1))
    intercept <- torch_ones(batch_size, 1L, dtype = h$dtype, device = h$device)
    h_input <- torch_cat(list(intercept, h_flat), dim = 2)
    self$pred(h_input)$squeeze(dim = 2)
  }
)

nam_midas_module <- nn_module(
  "NAM_MIDAS",

  initialize = function(d, K, L = 3L,
                        H_agg = 16L, n_agg_layers = 1L,
                        H_pred = 8L, n_pred_layers = 1L,
                        p_dropout = 0.2, p_group_dropout = 0.1,
                        use_neural_agg = TRUE,
                        use_residual = FALSE) {
    self$K <- as.integer(K)
    self$L <- as.integer(L)
    self$p_group_dropout <- p_group_dropout
    self$use_neural_agg <- use_neural_agg
    self$use_residual <- use_residual

    if (use_neural_agg || use_residual) {
      self$agg_nets <- nn_module_list(
        lapply(seq_len(K), function(k) {
          single_agg_net(as.integer(d), as.integer(L),
                         hidden_size = as.integer(H_agg),
                         n_layers = as.integer(n_agg_layers))
        })
      )
    }

    if (use_residual) {
      self$gamma_raw <- nn_parameter(torch_full(K, -2.0))
    }

    self$pred_nets <- nn_module_list(
      lapply(seq_len(K), function(k) {
        single_pred_net(as.integer(L),
                        hidden_size = as.integer(H_pred),
                        n_layers = as.integer(n_pred_layers),
                        p_dropout = p_dropout)
      })
    )

    self$bias <- nn_parameter(torch_zeros(1L))
  },

  forward = function(Z_lags, W_fixed = NULL) {
    gammas <- if (self$use_residual) torch_sigmoid(self$gamma_raw) else NULL
    contrib_list <- vector("list", self$K)

    for (k in seq_len(self$K)) {
      if (self$use_residual) {
        h_fixed <- torch_mm(Z_lags[[k]], W_fixed)
        h_neural <- self$agg_nets[[k]](Z_lags[[k]])
        h_k <- (1 - gammas[k]) * h_fixed + gammas[k] * h_neural
      } else if (self$use_neural_agg) {
        h_k <- self$agg_nets[[k]](Z_lags[[k]])
      } else {
        h_k <- torch_mm(Z_lags[[k]], W_fixed)
      }
      contrib_list[[k]] <- self$pred_nets[[k]](h_k)$squeeze(dim = 2)
    }

    stacked <- torch_stack(contrib_list, dim = 2)
    if (self$training && self$p_group_dropout > 0) {
      stacked <- nn_dropout(p = self$p_group_dropout)(stacked)
    }
    stacked$sum(dim = 2) + self$bias
  },

  get_gammas = function() {
    if (!is.null(self$gamma_raw)) {
      as.numeric(torch_sigmoid(self$gamma_raw)$cpu())
    } else { NULL }
  }
)

fanam_midas_module <- nn_module(
  "FANAM_MIDAS",

  initialize = function(d, K, L = 3L, r = 3L,
                        H_agg = 16L, n_agg_layers = 1L,
                        H_pred = 8L, n_pred_layers = 1L,
                        H_factor = 16L, n_factor_layers = 1L,
                        p_dropout = 0.2, p_group_dropout = 0.1,
                        use_neural_agg = TRUE,
                        use_residual = FALSE) {
    self$K <- as.integer(K)
    self$L <- as.integer(L)
    self$r <- as.integer(r)
    self$p_group_dropout <- p_group_dropout
    self$use_neural_agg <- use_neural_agg
    self$use_residual <- use_residual

    if (use_neural_agg || use_residual) {
      self$agg_nets <- nn_module_list(
        lapply(seq_len(K), function(k) {
          single_agg_net(as.integer(d), as.integer(L),
                         hidden_size = as.integer(H_agg),
                         n_layers = as.integer(n_agg_layers))
        })
      )
    }

    if (use_residual) {
      self$gamma_raw <- nn_parameter(torch_full(K, -2.0))
    }

    self$factor_proj <- NULL

    H_factor <- as.integer(H_factor)
    if (as.integer(n_factor_layers) >= 2L) {
      h2f <- as.integer(max(floor(H_factor / 2), 4L))
      self$factor_net <- nn_sequential(
        nn_linear(as.integer(r), H_factor),
        nn_relu(),
        nn_dropout(p = p_dropout),
        nn_linear(H_factor, h2f),
        nn_relu(),
        nn_dropout(p = p_dropout),
        nn_linear(h2f, 1L)
      )
    } else {
      self$factor_net <- nn_sequential(
        nn_linear(as.integer(r), H_factor),
        nn_relu(),
        nn_dropout(p = p_dropout),
        nn_linear(H_factor, 1L)
      )
    }

    self$pred_nets <- nn_module_list(
      lapply(seq_len(K), function(k) {
        single_pred_net(as.integer(L),
                        hidden_size = as.integer(H_pred),
                        n_layers = as.integer(n_pred_layers),
                        p_dropout = p_dropout)
      })
    )

    self$bias <- nn_parameter(torch_zeros(1L))
  },

  set_factor_projection = function(P_matrix) {
    self$factor_proj <- torch_tensor(as.matrix(P_matrix), dtype = torch_float())
  },

  forward = function(Z_lags, W_fixed = NULL) {
    gammas <- if (self$use_residual) torch_sigmoid(self$gamma_raw) else NULL
    h_list <- vector("list", self$K)
    contrib_list <- vector("list", self$K)

    for (k in seq_len(self$K)) {
      if (self$use_residual) {
        h_fixed <- torch_mm(Z_lags[[k]], W_fixed)
        h_neural <- self$agg_nets[[k]](Z_lags[[k]])
        h_k <- (1 - gammas[k]) * h_fixed + gammas[k] * h_neural
      } else if (self$use_neural_agg) {
        h_k <- self$agg_nets[[k]](Z_lags[[k]])
      } else {
        h_k <- torch_mm(Z_lags[[k]], W_fixed)
      }
      h_list[[k]] <- h_k
      contrib_list[[k]] <- self$pred_nets[[k]](h_k)$squeeze(dim = 2)
    }

    h_stacked <- torch_cat(h_list, dim = 2)
    f_tilde <- torch_mm(h_stacked, self$factor_proj$to(device = h_stacked$device))
    factor_contrib <- self$factor_net(f_tilde)$squeeze(dim = 2)

    stacked <- torch_stack(contrib_list, dim = 2)
    if (self$training && self$p_group_dropout > 0) {
      stacked <- nn_dropout(p = self$p_group_dropout)(stacked)
    }

    factor_contrib + stacked$sum(dim = 2) + self$bias
  },

  get_gammas = function() {
    if (!is.null(self$gamma_raw)) {
      as.numeric(torch_sigmoid(self$gamma_raw)$cpu())
    } else { NULL }
  }
)

fa_midas_module <- nn_module(
  "FA_MIDAS",

  initialize = function(d, K, L = 3L, r = 3L,
                        H_factor = 16L, n_factor_layers = 1L,
                        p_dropout = 0.2) {
    self$K <- as.integer(K); self$L <- as.integer(L); self$r <- as.integer(r)
    self$factor_proj   <- NULL
    self$lin_intercept <- NULL
    self$lin_weights   <- NULL
    H_factor <- as.integer(H_factor)
    if (as.integer(n_factor_layers) >= 2L) {
      h2f <- as.integer(max(floor(H_factor / 2), 4L))
      self$factor_net <- nn_sequential(
        nn_linear(as.integer(r), H_factor), nn_relu(), nn_dropout(p = p_dropout),
        nn_linear(H_factor, h2f), nn_relu(), nn_dropout(p = p_dropout),
        nn_linear(h2f, 1L))
    } else {
      self$factor_net <- nn_sequential(
        nn_linear(as.integer(r), H_factor), nn_relu(), nn_dropout(p = p_dropout),
        nn_linear(H_factor, 1L))
    }
  },

  set_factor_projection = function(P_matrix) {
    self$factor_proj <- torch_tensor(as.matrix(P_matrix), dtype = torch_float())
  },

  set_linear_offset = function(intercept, weights) {
    self$lin_intercept <- torch_tensor(as.numeric(intercept), dtype = torch_float())
    self$lin_weights   <- torch_tensor(as.numeric(weights),   dtype = torch_float())
  },

  forward = function(Z_lags, W_fixed) {
    h_list <- vector("list", self$K)
    for (k in seq_len(self$K)) h_list[[k]] <- torch_mm(Z_lags[[k]], W_fixed)
    h_stacked <- torch_cat(h_list, dim = 2)

    lin_w <- self$lin_weights$to(device = h_stacked$device)
    lin_b <- self$lin_intercept$to(device = h_stacked$device)
    base_logit <- torch_mv(h_stacked, lin_w) + lin_b

    f_tilde <- torch_mm(h_stacked, self$factor_proj$to(device = h_stacked$device))
    factor_contrib <- self$factor_net(f_tilde)$squeeze(dim = 2)

    base_logit + factor_contrib
  }
)

ipcw_bce_loss <- function(logit, y, ipcw_weights) {
  loss_per_obs <- ipcw_weights * (
    torch_clamp(logit, min = 0) - logit * y + torch_log(1 + torch_exp(-torch_abs(logit)))
  )
  loss_per_obs$mean()
}

compute_nam_penalty <- function(model, K, lambda1 = 0.01, lambda2 = 0.01) {
  l1_pen <- torch_tensor(0, dtype = torch_float(), requires_grad = FALSE)
  l2_pen <- torch_tensor(0, dtype = torch_float(), requires_grad = FALSE)

  for (k in seq_len(K)) {
    W_k <- model$pred_nets[[k]]$fc1$weight
    l1_pen <- l1_pen + W_k$abs()$sum()
    l2_pen <- l2_pen + W_k$norm(2)
  }

  lambda1 * l1_pen + lambda2 * l2_pen
}

compute_output_penalty <- function(model, Z_lags, W_fixed = NULL, K) {
  model$eval()

  gammas <- NULL
  if (model$use_residual && !is.null(model$gamma_raw)) {
    gammas <- torch_sigmoid(model$gamma_raw)
  }

  output_sq_sum <- torch_tensor(0, dtype = torch_float(), requires_grad = FALSE)

  for (k in seq_len(K)) {
    if (!is.null(gammas)) {
      h_fixed <- torch_mm(Z_lags[[k]], W_fixed)
      h_neural <- model$agg_nets[[k]](Z_lags[[k]])
      h_k <- (1 - gammas[k]) * h_fixed + gammas[k] * h_neural
    } else if (model$use_neural_agg) {
      h_k <- model$agg_nets[[k]](Z_lags[[k]])
    } else {
      h_k <- torch_mm(Z_lags[[k]], W_fixed)
    }

    f_k_out <- model$pred_nets[[k]](h_k)$squeeze(dim = 2)
    output_sq_sum <- output_sq_sum + torch_mean(f_k_out^2)
  }

  model$train()
  output_sq_sum / K
}

compute_factor_projection <- function(Z_lags_train, K, L, r, W_fixed,
                                      rotate = TRUE) {
  n <- nrow(Z_lags_train[[1]])
  H_mat <- matrix(0, nrow = n, ncol = K * L)

  for (k in seq_len(K)) {
    col_start <- (k - 1L) * L + 1L
    col_end <- k * L
    H_mat[, col_start:col_end] <- as.matrix(Z_lags_train[[k]]) %*% as.matrix(W_fixed)
  }

  col_sds <- apply(H_mat, 2, sd)
  col_sds[col_sds < 1e-8] <- 1
  H_std <- scale(H_mat, center = TRUE, scale = col_sds)

  cov_mat <- crossprod(H_std) / n
  trace_cov <- sum(diag(cov_mat))
  r_use <- min(r, ncol(cov_mat) - 1, n - 1)
  if (r_use < 1) r_use <- 1L

  eig <- tryCatch({
    RSpectra::eigs_sym(cov_mat, k = r_use, which = "LM")
  }, error = function(e) {
    e_full <- eigen(cov_mat, symmetric = TRUE)
    list(vectors = e_full$vectors[, 1:r_use, drop = FALSE],
         values = e_full$values[1:r_use])
  })

  V <- eig$vectors[, seq_len(r_use), drop = FALSE]
  lam <- pmax(eig$values[seq_len(r_use)], 0)
  ev <- lam / trace_cov

  if (isTRUE(rotate) && r_use >= 2L) {
    rot <- tryCatch(stats::varimax(V), error = function(e) NULL)
    if (!is.null(rot)) {
      V <- as.matrix(rot$loadings)
      new_lam <- diag(t(V) %*% cov_mat %*% V)
      ev <- pmax(new_lam, 0) / trace_cov
    }
  }

  P_scaled <- sweep(V, 1, col_sds, "/")

  list(P = P_scaled,
       explained_var = ev,
       r_used = r_use)
}

compute_group_penalty <- function(model, K, L, lambda1 = 0.01, lambda2 = 0.01) {
  W_first <- model$pred[[1]]$weight
  W_groups <- W_first[, 2:(K * L + 1)]
  l1_pen <- torch_tensor(0, dtype = W_groups$dtype, device = W_groups$device, requires_grad = FALSE)
  l2_pen <- torch_tensor(0, dtype = W_groups$dtype, device = W_groups$device, requires_grad = FALSE)
  for (k in seq_len(K)) {
    col_start <- (k - 1L) * L + 1L
    col_end <- k * L
    W_k <- W_groups[, col_start:col_end]
    l1_pen <- l1_pen + W_k$abs()$sum()
    l2_pen <- l2_pen + W_k$norm(2)
  }
  lambda1 * l1_pen + lambda2 * l2_pen
}

standardise_lag_groups <- function(Z_lags_train, Z_lags_test = NULL) {
  K <- length(Z_lags_train)
  Z_train_std <- vector("list", K)
  Z_test_std  <- if (!is.null(Z_lags_test)) vector("list", K) else NULL
  for (k in seq_len(K)) {
    m <- colMeans(Z_lags_train[[k]], na.rm = TRUE)
    s <- apply(Z_lags_train[[k]], 2, sd, na.rm = TRUE)
    s[s < 1e-8] <- 1
    Z_train_std[[k]] <- sweep(sweep(Z_lags_train[[k]], 2, m, "-"), 2, s, "/")
    if (!is.null(Z_lags_test)) {
      Z_test_std[[k]] <- sweep(sweep(Z_lags_test[[k]], 2, m, "-"), 2, s, "/")
    }
  }
  list(train = Z_train_std, test = Z_test_std)
}

stratified_split <- function(y, val_frac = 0.2) {
  pos_idx <- which(y == 1)
  neg_idx <- which(y == 0)
  n_pos_val <- min(max(round(length(pos_idx) * val_frac), 2), length(pos_idx) - 2)
  n_neg_val <- min(max(round(length(neg_idx) * val_frac), 2), length(neg_idx) - 2)
  val_pos <- sample(pos_idx, n_pos_val)
  val_neg <- sample(neg_idx, n_neg_val)
  list(train = setdiff(seq_along(y), c(val_pos, val_neg)), val = c(val_pos, val_neg))
}

train_nam_midas <- function(Z_lags_train, y_train, w_train,
                            Z_lags_val, y_val, w_val,
                            K, d, L = 3L,
                            H_agg = 16L, n_agg_layers = 1L,
                            H_pred = 8L, n_pred_layers = 1L,
                            p_dropout = 0.2, p_group_dropout = 0.1,
                            use_neural_agg = TRUE,
                            use_residual = FALSE,
                            W_fixed = NULL,
                            lr = 1e-3, weight_decay = 1e-4,
                            lambda1 = 0.01, lambda2 = 0.01,
                            output_reg = 0.0,
                            n_epochs = 500L, patience = 40L,
                            grad_clip = 1.0, verbose = FALSE) {

  stopifnot(is.list(Z_lags_train), length(Z_lags_train) == K)
  if (!use_neural_agg || use_residual) stopifnot(!is.null(W_fixed))

  Z_train_t <- lapply(Z_lags_train, function(x) torch_tensor(as.matrix(x), dtype = torch_float()))
  Z_val_t   <- lapply(Z_lags_val, function(x) torch_tensor(as.matrix(x), dtype = torch_float()))
  y_train_t <- torch_tensor(as.numeric(y_train), dtype = torch_float())
  y_val_t   <- torch_tensor(as.numeric(y_val), dtype = torch_float())
  w_train_t <- torch_tensor(as.numeric(w_train), dtype = torch_float())
  w_val_t   <- torch_tensor(as.numeric(w_val), dtype = torch_float())
  W_fixed_t <- if (!is.null(W_fixed)) torch_tensor(as.matrix(W_fixed), dtype = torch_float()) else NULL

  model <- nam_midas_module(
    d = as.integer(d), K = as.integer(K), L = as.integer(L),
    H_agg = as.integer(H_agg), n_agg_layers = as.integer(n_agg_layers),
    H_pred = as.integer(H_pred), n_pred_layers = as.integer(n_pred_layers),
    p_dropout = p_dropout, p_group_dropout = p_group_dropout,
    use_neural_agg = use_neural_agg,
    use_residual = use_residual
  )

  optimizer <- optim_adam(model$parameters, lr = lr, weight_decay = weight_decay)
  K_int <- as.integer(K)
  best_val_loss <- Inf; best_epoch <- 0L; best_state <- NULL

  for (epoch in seq_len(n_epochs)) {
    model$train()
    optimizer$zero_grad()
    logit_train <- model(Z_train_t, W_fixed_t)
    loss_train <- ipcw_bce_loss(logit_train, y_train_t, w_train_t)
    penalty <- compute_nam_penalty(model, K_int, lambda1, lambda2)
    out_pen <- if (output_reg > 0) {
      output_reg * compute_output_penalty(model, Z_train_t, W_fixed_t, K_int)
    } else { torch_tensor(0, dtype = torch_float()) }
    total_loss <- loss_train + penalty + out_pen
    total_loss$backward()
    if (grad_clip > 0) nn_utils_clip_grad_norm_(model$parameters, max_norm = grad_clip)
    optimizer$step()

    model$eval()
    val_loss_value <- with_no_grad({
      ipcw_bce_loss(model(Z_val_t, W_fixed_t), y_val_t, w_val_t)$item()
    })

    if (val_loss_value < best_val_loss) {
      best_val_loss <- val_loss_value; best_epoch <- epoch
      best_state <- lapply(model$state_dict(), function(x) x$clone())
    }
    if (epoch - best_epoch >= patience) break
  }

  if (!is.null(best_state)) model$load_state_dict(best_state)
  gammas <- if (use_residual) model$get_gammas() else NULL

  list(model = model, best_epoch = best_epoch, best_val_loss = best_val_loss,
       gammas = gammas)
}

train_fanam_midas <- function(Z_lags_train, y_train, w_train,
                              Z_lags_val, y_val, w_val,
                              K, d, L = 3L, r = 3L,
                              H_agg = 16L, n_agg_layers = 1L,
                              H_pred = 8L, n_pred_layers = 1L,
                              H_factor = 16L, n_factor_layers = 1L,
                              p_dropout = 0.2, p_group_dropout = 0.1,
                              use_neural_agg = TRUE,
                              use_residual = FALSE,
                              W_fixed = NULL,
                              lr = 1e-3, weight_decay = 1e-4,
                              lambda1 = 0.01, lambda2 = 0.01,
                              output_reg = 0.0,
                              n_epochs = 500L, patience = 40L,
                              grad_clip = 1.0, verbose = FALSE) {

  stopifnot(is.list(Z_lags_train), length(Z_lags_train) == K)
  if (!use_neural_agg || use_residual) stopifnot(!is.null(W_fixed))

  pca_res <- compute_factor_projection(Z_lags_train, K = K, L = L, r = r, W_fixed = W_fixed)

  Z_train_t <- lapply(Z_lags_train, function(x) torch_tensor(as.matrix(x), dtype = torch_float()))
  Z_val_t   <- lapply(Z_lags_val, function(x) torch_tensor(as.matrix(x), dtype = torch_float()))
  y_train_t <- torch_tensor(as.numeric(y_train), dtype = torch_float())
  y_val_t   <- torch_tensor(as.numeric(y_val), dtype = torch_float())
  w_train_t <- torch_tensor(as.numeric(w_train), dtype = torch_float())
  w_val_t   <- torch_tensor(as.numeric(w_val), dtype = torch_float())
  W_fixed_t <- if (!is.null(W_fixed)) torch_tensor(as.matrix(W_fixed), dtype = torch_float()) else NULL

  model <- fanam_midas_module(
    d = as.integer(d), K = as.integer(K), L = as.integer(L),
    r = as.integer(pca_res$r_used),
    H_agg = as.integer(H_agg), n_agg_layers = as.integer(n_agg_layers),
    H_pred = as.integer(H_pred), n_pred_layers = as.integer(n_pred_layers),
    H_factor = as.integer(H_factor), n_factor_layers = as.integer(n_factor_layers),
    p_dropout = p_dropout, p_group_dropout = p_group_dropout,
    use_neural_agg = use_neural_agg,
    use_residual = use_residual
  )

  model$set_factor_projection(pca_res$P)

  optimizer <- optim_adam(model$parameters, lr = lr, weight_decay = weight_decay)
  K_int <- as.integer(K)
  best_val_loss <- Inf; best_epoch <- 0L; best_state <- NULL

  for (epoch in seq_len(n_epochs)) {
    model$train()
    optimizer$zero_grad()
    logit_train <- model(Z_train_t, W_fixed_t)
    loss_train <- ipcw_bce_loss(logit_train, y_train_t, w_train_t)
    penalty <- compute_nam_penalty(model, K_int, lambda1, lambda2)
    out_pen <- if (output_reg > 0) {
      output_reg * compute_output_penalty(model, Z_train_t, W_fixed_t, K_int)
    } else { torch_tensor(0, dtype = torch_float()) }
    total_loss <- loss_train + penalty + out_pen
    total_loss$backward()
    if (grad_clip > 0) nn_utils_clip_grad_norm_(model$parameters, max_norm = grad_clip)
    optimizer$step()

    model$eval()
    val_loss_value <- with_no_grad({
      ipcw_bce_loss(model(Z_val_t, W_fixed_t), y_val_t, w_val_t)$item()
    })

    if (val_loss_value < best_val_loss) {
      best_val_loss <- val_loss_value; best_epoch <- epoch
      best_state <- lapply(model$state_dict(), function(x) x$clone())
    }
    if (epoch - best_epoch >= patience) break
  }

  if (!is.null(best_state)) model$load_state_dict(best_state)
  gammas <- if (use_residual) model$get_gammas() else NULL

  list(model = model, best_epoch = best_epoch, best_val_loss = best_val_loss,
       gammas = gammas, pca_res = pca_res)
}

train_fa_midas <- function(Z_lags_train, y_train, w_train,
                           Z_lags_val, y_val, w_val,
                           K, d, L = 3L, r = 3L,
                           H_factor = 16L, n_factor_layers = 1L,
                           p_dropout = 0.2,
                           W_fixed = NULL,
                           lin_intercept = 0, lin_weights = NULL,
                           lr = 1e-3, weight_decay = 1e-4,
                           n_epochs = 500L, patience = 40L,
                           grad_clip = 1.0, verbose = FALSE) {

  stopifnot(is.list(Z_lags_train), length(Z_lags_train) == K)
  stopifnot(!is.null(W_fixed), !is.null(lin_weights))
  stopifnot(length(lin_weights) == K * L)

  pca_res <- compute_factor_projection(Z_lags_train, K = K, L = L, r = r, W_fixed = W_fixed)

  Z_train_t <- lapply(Z_lags_train, function(x) torch_tensor(as.matrix(x), dtype = torch_float()))
  Z_val_t   <- lapply(Z_lags_val, function(x) torch_tensor(as.matrix(x), dtype = torch_float()))
  y_train_t <- torch_tensor(as.numeric(y_train), dtype = torch_float())
  y_val_t   <- torch_tensor(as.numeric(y_val), dtype = torch_float())
  w_train_t <- torch_tensor(as.numeric(w_train), dtype = torch_float())
  w_val_t   <- torch_tensor(as.numeric(w_val), dtype = torch_float())
  W_fixed_t <- torch_tensor(as.matrix(W_fixed), dtype = torch_float())

  model <- fa_midas_module(
    d = as.integer(d), K = as.integer(K), L = as.integer(L),
    r = as.integer(pca_res$r_used),
    H_factor = as.integer(H_factor), n_factor_layers = as.integer(n_factor_layers),
    p_dropout = p_dropout
  )

  model$set_factor_projection(pca_res$P)
  model$set_linear_offset(lin_intercept, lin_weights)

  optimizer <- optim_adam(model$factor_net$parameters, lr = lr, weight_decay = weight_decay)
  best_val_loss <- Inf; best_epoch <- 0L; best_state <- NULL

  for (epoch in seq_len(n_epochs)) {
    model$train()
    optimizer$zero_grad()
    logit_train <- model(Z_train_t, W_fixed_t)
    loss_train <- ipcw_bce_loss(logit_train, y_train_t, w_train_t)
    loss_train$backward()
    if (grad_clip > 0) nn_utils_clip_grad_norm_(model$factor_net$parameters, max_norm = grad_clip)
    optimizer$step()

    model$eval()
    val_loss_value <- with_no_grad({
      ipcw_bce_loss(model(Z_val_t, W_fixed_t), y_val_t, w_val_t)$item()
    })

    if (val_loss_value < best_val_loss) {
      best_val_loss <- val_loss_value; best_epoch <- epoch
      best_state <- lapply(model$state_dict(), function(x) x$clone())
    }
    if (epoch - best_epoch >= patience) break
  }

  if (!is.null(best_state)) model$load_state_dict(best_state)

  list(model = model, best_epoch = best_epoch, best_val_loss = best_val_loss,
       pca_res = pca_res)
}

train_deep_midas <- function(Z_lags_train, y_train, w_train,
                             Z_lags_val, y_val, w_val,
                             K, d, L = 3L, H_agg = 16L, H_pred = 32L,
                             n_pred_layers = 1L, n_agg_layers = 1L,
                             p_dropout = 0.2, p_group_dropout = 0.1,
                             use_neural_agg = TRUE,
                             W_fixed = NULL,
                             lr = 1e-3, weight_decay = 1e-4,
                             lambda1 = 0.01, lambda2 = 0.01,
                             n_epochs = 500L, patience = 40L,
                             grad_clip = 1.0, verbose = FALSE) {

  stopifnot(is.list(Z_lags_train), length(Z_lags_train) == K)
  stopifnot(length(y_train) == nrow(Z_lags_train[[1]]))
  if (!use_neural_agg) stopifnot(!is.null(W_fixed))

  Z_train_t <- lapply(Z_lags_train, function(x) torch_tensor(as.matrix(x), dtype = torch_float()))
  Z_val_t   <- lapply(Z_lags_val, function(x) torch_tensor(as.matrix(x), dtype = torch_float()))
  y_train_t <- torch_tensor(as.numeric(y_train), dtype = torch_float())
  y_val_t   <- torch_tensor(as.numeric(y_val), dtype = torch_float())
  w_train_t <- torch_tensor(as.numeric(w_train), dtype = torch_float())
  w_val_t   <- torch_tensor(as.numeric(w_val), dtype = torch_float())
  W_fixed_t <- if (!is.null(W_fixed)) torch_tensor(as.matrix(W_fixed), dtype = torch_float()) else NULL

  model <- deep_midas_module(
    d = as.integer(d), K = as.integer(K), L = as.integer(L),
    H_agg = as.integer(H_agg), H_pred = as.integer(H_pred),
    n_pred_layers = as.integer(n_pred_layers),
    n_agg_layers = as.integer(n_agg_layers),
    p_dropout = p_dropout, p_group_dropout = p_group_dropout,
    use_neural_agg = use_neural_agg
  )

  optimizer <- optim_adam(model$parameters, lr = lr, weight_decay = weight_decay)
  K_int <- as.integer(K); L_int <- as.integer(L)
  best_val_loss <- Inf; best_epoch <- 0L; best_state <- NULL

  for (epoch in seq_len(n_epochs)) {
    model$train()
    optimizer$zero_grad()
    logit_train <- model(Z_train_t, W_fixed_t)
    loss_train <- ipcw_bce_loss(logit_train, y_train_t, w_train_t)
    penalty <- compute_group_penalty(model, K_int, L_int, lambda1, lambda2)
    total_loss <- loss_train + penalty
    total_loss$backward()
    if (grad_clip > 0) nn_utils_clip_grad_norm_(model$parameters, max_norm = grad_clip)
    optimizer$step()

    model$eval()
    val_loss_value <- with_no_grad({
      lv <- ipcw_bce_loss(model(Z_val_t, W_fixed_t), y_val_t, w_val_t)
      lv$item()
    })

    if (val_loss_value < best_val_loss) {
      best_val_loss <- val_loss_value; best_epoch <- epoch
      best_state <- lapply(model$state_dict(), function(x) x$clone())
    }
    if (epoch - best_epoch >= patience) break
  }

  if (!is.null(best_state)) model$load_state_dict(best_state)

  list(model = model, best_epoch = best_epoch, best_val_loss = best_val_loss,
       gamma = NA)
}

predict_nam_midas <- function(model, Z_lags_test, W_fixed = NULL) {
  Z_test_t <- lapply(Z_lags_test, function(x) torch_tensor(as.matrix(x), dtype = torch_float()))
  W_fixed_t <- if (!is.null(W_fixed)) torch_tensor(as.matrix(W_fixed), dtype = torch_float()) else NULL
  model$eval()
  preds <- with_no_grad({ torch_sigmoid(model(Z_test_t, W_fixed_t)) })
  as.numeric(preds$cpu())
}

predict_deep_midas <- function(model, Z_lags_test, W_fixed = NULL) {
  Z_test_t <- lapply(Z_lags_test, function(x) torch_tensor(as.matrix(x), dtype = torch_float()))
  W_fixed_t <- if (!is.null(W_fixed)) torch_tensor(as.matrix(W_fixed), dtype = torch_float()) else NULL
  model$eval()
  preds <- with_no_grad({ torch_sigmoid(model(Z_test_t, W_fixed_t)) })
  as.numeric(preds$cpu())
}

predict_fa_midas <- function(model, Z_lags_test, W_fixed) {
  Z_test_t  <- lapply(Z_lags_test, function(x) torch_tensor(as.matrix(x), dtype = torch_float()))
  W_fixed_t <- torch_tensor(as.matrix(W_fixed), dtype = torch_float())
  model$eval()
  preds <- with_no_grad({ torch_sigmoid(model(Z_test_t, W_fixed_t)) })
  as.numeric(preds$cpu())
}

extract_contributions <- function(model, Z_lags, W_fixed = NULL, is_fanam = FALSE) {
  Z_t <- lapply(Z_lags, function(x) torch_tensor(as.matrix(x), dtype = torch_float()))
  W_fixed_t <- if (!is.null(W_fixed)) torch_tensor(as.matrix(W_fixed), dtype = torch_float()) else NULL

  model$eval()
  K <- model$K

  with_no_grad({
    gammas <- if (model$use_residual && !is.null(model$gamma_raw)) {
      torch_sigmoid(model$gamma_raw)
    } else { NULL }

    h_list <- vector("list", K)
    contrib_list <- vector("list", K)

    for (k in seq_len(K)) {
      if (model$use_residual && !is.null(gammas)) {
        h_fixed <- torch_mm(Z_t[[k]], W_fixed_t)
        h_neural <- model$agg_nets[[k]](Z_t[[k]])
        h_k <- (1 - gammas[k]) * h_fixed + gammas[k] * h_neural
      } else if (model$use_neural_agg) {
        h_k <- model$agg_nets[[k]](Z_t[[k]])
      } else {
        h_k <- torch_mm(Z_t[[k]], W_fixed_t)
      }
      h_list[[k]] <- as.matrix(h_k$cpu())
      contrib_list[[k]] <- as.numeric(model$pred_nets[[k]](h_k)$squeeze(dim = 2)$cpu())
    }

    contrib_matrix <- do.call(cbind, contrib_list)

    h_matrix <- do.call(cbind, h_list)

    factor_contrib_vec <- NULL
    if (is_fanam && !is.null(model$factor_proj)) {
      h_stacked <- torch_tensor(h_matrix, dtype = torch_float())
      f_tilde <- torch_mm(h_stacked, model$factor_proj$to(device = h_stacked$device))
      factor_contrib_vec <- as.numeric(model$factor_net(f_tilde)$squeeze(dim = 2)$cpu())
    }

    bias_val <- as.numeric(model$bias$cpu())
  })

  list(
    contrib_matrix = contrib_matrix,
    h_matrix = h_matrix,
    factor_contrib = factor_contrib_vec,
    bias = bias_val
  )
}

extract_lag_groups <- function(X_matrix, num_vars, jmax) {
  lag_groups <- vector("list", num_vars)
  for (k in seq_len(num_vars)) {
    col_start <- (k - 1L) * jmax + 1L
    col_end <- k * jmax
    lag_groups[[k]] <- X_matrix[, col_start:col_end, drop = FALSE]
  }
  lag_groups
}

cv_nam_midas <- function(Z_lags, y, w, K, d, jmax,
                         nfolds = 3, n_random = 50L,
                         use_neural_agg = TRUE, use_residual = FALSE,
                         use_factor = FALSE,
                         W_fixed = NULL,
                         data_for_auc = NULL, t_horizon = 0) {
  N <- length(y)
  pos_idx <- which(y == 1); neg_idx <- which(y == 0)
  foldid <- integer(N)
  foldid[pos_idx] <- sample(rep(1:nfolds, length.out = length(pos_idx)))
  foldid[neg_idx] <- sample(rep(1:nfolds, length.out = length(neg_idx)))

  param_grid <- data.frame(
    lr              = sample(c(1e-4, 5e-4, 1e-3, 3e-3), n_random, replace = TRUE),
    L               = sample(c(2L, 3L, 4L, 5L), n_random, replace = TRUE),
    H_agg           = sample(c(8L, 16L, 32L, 64L, 128L), n_random, replace = TRUE),
    n_agg_layers    = sample(c(1L, 2L, 3L, 4L), n_random, replace = TRUE),
    H_pred          = sample(c(4L, 8L, 16L, 32L), n_random, replace = TRUE),
    n_pred_layers   = sample(c(1L, 2L, 3L, 4L), n_random, replace = TRUE),
    lambda1         = sample(c(0, 0.001, 0.01, 0.1, 0.5), n_random, replace = TRUE),
    lambda2         = sample(c(0, 0.001, 0.01, 0.1, 0.5), n_random, replace = TRUE),
    p_dropout       = sample(c(0, 0.1, 0.2, 0.4, 0.5), n_random, replace = TRUE),
    p_group_dropout = sample(c(0, 0.1, 0.2, 0.3, 0.5), n_random, replace = TRUE),
    weight_decay    = sample(c(1e-5, 1e-4, 1e-3, 1e-2), n_random, replace = TRUE),
    output_reg      = sample(c(0, 0.001, 0.01, 0.1), n_random, replace = TRUE),
    patience        = sample(c(20L, 30L, 40L, 50L), n_random, replace = TRUE),
    grad_clip       = sample(c(0.5, 1.0, 2.0, 5.0), n_random, replace = TRUE),
    stringsAsFactors = FALSE
  )

  if (use_factor) {
    param_grid$r              <- sample(c(2L, 3L, 4L, 5L), n_random, replace = TRUE)
    param_grid$H_factor       <- sample(c(8L, 16L, 32L, 64L), n_random, replace = TRUE)
    param_grid$n_factor_layers <- sample(c(1L, 2L), n_random, replace = TRUE)
  }

  param_grid <- unique(param_grid)
  best_auc <- -Inf; best_params <- NULL

  for (row in seq_len(nrow(param_grid))) {
    params <- param_grid[row, ]
    fold_aucs <- c()

    L_cv <- as.integer(params$L)
    W_fixed_cv <- gb(degree = L_cv - 1L, alpha = -1/2, a = 0, b = 1, jmax = jmax) / jmax

    for (fold in seq_len(nfolds)) {
      val_idx <- which(foldid == fold)
      train_idx <- which(foldid != fold)
      Z_tr <- lapply(Z_lags, function(x) x[train_idx, , drop = FALSE])
      Z_vl <- lapply(Z_lags, function(x) x[val_idx, , drop = FALSE])
      std <- standardise_lag_groups(Z_tr, Z_vl)

      fit <- tryCatch({
        if (use_factor) {
          train_fanam_midas(
            std$train, y[train_idx], w[train_idx],
            std$test, y[val_idx], w[val_idx],
            K = K, d = d, L = L_cv, r = params$r,
            H_agg = params$H_agg, n_agg_layers = params$n_agg_layers,
            H_pred = params$H_pred, n_pred_layers = params$n_pred_layers,
            H_factor = params$H_factor, n_factor_layers = params$n_factor_layers,
            p_dropout = params$p_dropout, p_group_dropout = params$p_group_dropout,
            use_neural_agg = use_neural_agg, use_residual = use_residual,
            W_fixed = W_fixed_cv,
            lr = params$lr, weight_decay = params$weight_decay,
            lambda1 = params$lambda1, lambda2 = params$lambda2,
            output_reg = params$output_reg,
            n_epochs = 300L, patience = params$patience,
            grad_clip = params$grad_clip
          )
        } else {
          train_nam_midas(
            std$train, y[train_idx], w[train_idx],
            std$test, y[val_idx], w[val_idx],
            K = K, d = d, L = L_cv,
            H_agg = params$H_agg, n_agg_layers = params$n_agg_layers,
            H_pred = params$H_pred, n_pred_layers = params$n_pred_layers,
            p_dropout = params$p_dropout, p_group_dropout = params$p_group_dropout,
            use_neural_agg = use_neural_agg, use_residual = use_residual,
            W_fixed = W_fixed_cv,
            lr = params$lr, weight_decay = params$weight_decay,
            lambda1 = params$lambda1, lambda2 = params$lambda2,
            output_reg = params$output_reg,
            n_epochs = 300L, patience = params$patience,
            grad_clip = params$grad_clip
          )
        }
      }, error = function(e) NULL)

      if (is.null(fit)) next
      val_preds <- predict_nam_midas(fit$model, std$test, W_fixed_cv)

      if (!is.null(data_for_auc)) {
        auc_val <- tryCatch({
          survivalROC(Stime = data_for_auc$time[val_idx],
                      status = data_for_auc$status[val_idx],
                      marker = val_preds, predict.time = t_horizon,
                      span = 0.25 * length(val_idx)^(-1/2))$AUC
        }, error = function(e) NA)
        if (!is.na(auc_val)) fold_aucs <- c(fold_aucs, auc_val)
      }
    }

    if (length(fold_aucs) >= 2) {
      mean_auc <- mean(fold_aucs)
      if (mean_auc > best_auc) { best_auc <- mean_auc; best_params <- params }
    }
  }
  list(best_params = best_params, best_cv_auc = best_auc)
}

cv_deep_midas <- function(Z_lags, y, w, K, d, jmax,
                          nfolds = 3, n_random = 50L,
                          use_neural_agg = TRUE,
                          W_fixed = NULL,
                          data_for_auc = NULL, t_horizon = 0) {
  N <- length(y)
  pos_idx <- which(y == 1); neg_idx <- which(y == 0)
  foldid <- integer(N)
  foldid[pos_idx] <- sample(rep(1:nfolds, length.out = length(pos_idx)))
  foldid[neg_idx] <- sample(rep(1:nfolds, length.out = length(neg_idx)))

  param_grid <- data.frame(
    lr              = sample(c(1e-4, 5e-4, 1e-3, 3e-3), n_random, replace = TRUE),
    L               = sample(c(2L, 3L, 4L, 5L), n_random, replace = TRUE),
    H_agg           = sample(c(8L, 16L, 32L, 64L, 128L), n_random, replace = TRUE),
    H_pred          = sample(c(16L, 32L, 64L, 128L), n_random, replace = TRUE),
    n_pred_layers   = sample(c(1L, 2L, 3L, 4L), n_random, replace = TRUE),
    n_agg_layers    = sample(c(1L, 2L, 3L, 4L), n_random, replace = TRUE),
    lambda1         = sample(c(0, 0.001, 0.01, 0.1, 0.5), n_random, replace = TRUE),
    lambda2         = sample(c(0, 0.001, 0.01, 0.1, 0.5), n_random, replace = TRUE),
    p_dropout       = sample(c(0, 0.1, 0.2, 0.4, 0.5), n_random, replace = TRUE),
    p_group_dropout = sample(c(0, 0.1, 0.2, 0.3, 0.5), n_random, replace = TRUE),
    weight_decay    = sample(c(1e-5, 1e-4, 1e-3, 1e-2), n_random, replace = TRUE),
    patience        = sample(c(20L, 30L, 40L, 50L), n_random, replace = TRUE),
    grad_clip       = sample(c(0.5, 1.0, 2.0, 5.0), n_random, replace = TRUE),
    stringsAsFactors = FALSE
  )
  param_grid <- unique(param_grid)

  best_auc <- -Inf; best_params <- NULL

  for (row in seq_len(nrow(param_grid))) {
    params <- param_grid[row, ]
    fold_aucs <- c()

    L_cv <- as.integer(params$L)
    W_fixed_cv <- gb(degree = L_cv - 1L, alpha = -1/2, a = 0, b = 1, jmax = jmax) / jmax

    W_fixed_for_call <- if (!use_neural_agg) W_fixed_cv else NULL

    for (fold in seq_len(nfolds)) {
      val_idx <- which(foldid == fold); train_idx <- which(foldid != fold)
      Z_tr <- lapply(Z_lags, function(x) x[train_idx, , drop = FALSE])
      Z_vl <- lapply(Z_lags, function(x) x[val_idx, , drop = FALSE])
      std <- standardise_lag_groups(Z_tr, Z_vl)

      fit <- tryCatch({
        train_deep_midas(
          std$train, y[train_idx], w[train_idx],
          std$test, y[val_idx], w[val_idx],
          K = K, d = d, L = L_cv,
          H_agg = params$H_agg, H_pred = params$H_pred,
          n_pred_layers = params$n_pred_layers,
          n_agg_layers = params$n_agg_layers,
          p_dropout = params$p_dropout, p_group_dropout = params$p_group_dropout,
          use_neural_agg = use_neural_agg,
          W_fixed = W_fixed_for_call,
          lr = params$lr, weight_decay = params$weight_decay,
          lambda1 = params$lambda1, lambda2 = params$lambda2,
          n_epochs = 300L, patience = params$patience,
          grad_clip = params$grad_clip, verbose = FALSE
        )
      }, error = function(e) NULL)

      if (is.null(fit)) next
      val_preds <- predict_deep_midas(fit$model, std$test, W_fixed_for_call)

      if (!is.null(data_for_auc)) {
        auc_val <- tryCatch({
          survivalROC(Stime = data_for_auc$time[val_idx],
                      status = data_for_auc$status[val_idx],
                      marker = val_preds, predict.time = t_horizon,
                      span = 0.25 * length(val_idx)^(-1/2))$AUC
        }, error = function(e) NA)
        if (!is.na(auc_val)) fold_aucs <- c(fold_aucs, auc_val)
      }
    }
    if (length(fold_aucs) >= 2) {
      mean_auc <- mean(fold_aucs)
      if (mean_auc > best_auc) { best_auc <- mean_auc; best_params <- params }
    }
  }
  list(best_params = best_params, best_cv_auc = best_auc)
}

cv_fa_midas <- function(Z_lags, y, w, K, d, jmax,
                        lin_intercept, lin_weights,
                        nfolds = 3, n_random = 50L,
                        W_fixed = NULL,
                        data_for_auc = NULL, t_horizon = 0) {
  N <- length(y)
  pos_idx <- which(y == 1); neg_idx <- which(y == 0)
  foldid <- integer(N)
  foldid[pos_idx] <- sample(rep(1:nfolds, length.out = length(pos_idx)))
  foldid[neg_idx] <- sample(rep(1:nfolds, length.out = length(neg_idx)))

  param_grid <- data.frame(
    lr              = sample(c(1e-4, 5e-4, 1e-3, 3e-3), n_random, replace = TRUE),
    L               = sample(c(2L, 3L, 4L, 5L),         n_random, replace = TRUE),
    r               = sample(c(2L, 3L, 4L, 5L),         n_random, replace = TRUE),
    H_factor        = sample(c(8L, 16L, 32L, 64L),      n_random, replace = TRUE),
    n_factor_layers = sample(c(1L, 2L),                  n_random, replace = TRUE),
    p_dropout       = sample(c(0, 0.1, 0.2, 0.4, 0.5),  n_random, replace = TRUE),
    weight_decay    = sample(c(1e-5, 1e-4, 1e-3, 1e-2), n_random, replace = TRUE),
    patience        = sample(c(20L, 30L, 40L, 50L),      n_random, replace = TRUE),
    grad_clip       = sample(c(0.5, 1.0, 2.0, 5.0),      n_random, replace = TRUE),
    stringsAsFactors = FALSE
  )
  param_grid <- unique(param_grid)

  best_auc <- -Inf; best_params <- NULL

  for (row in seq_len(nrow(param_grid))) {
    params <- param_grid[row, ]
    fold_aucs <- c()

    L_cv <- as.integer(params$L)
    W_fixed_cv <- gb(degree = L_cv - 1L, alpha = -1/2, a = 0, b = 1, jmax = jmax) / jmax

    if (L_cv == length(lin_weights) / K) {
      lin_int_cv <- lin_intercept
      lin_w_cv   <- lin_weights
    } else {
      H_full <- matrix(0, nrow = N, ncol = K * L_cv)
      for (k in seq_len(K)) {
        H_full[, ((k - 1L) * L_cv + 1L):(k * L_cv)] <-
          as.matrix(Z_lags[[k]]) %*% as.matrix(W_fixed_cv)
      }
      lin_fit <- tryCatch({
        suppressWarnings(glm.fit(x = cbind(1, H_full), y = y, weights = w,
                                 family = binomial()))
      }, error = function(e) NULL)
      if (is.null(lin_fit) || any(!is.finite(lin_fit$coefficients))) {
        lin_int_cv <- 0; lin_w_cv <- rep(0, K * L_cv)
      } else {
        lin_int_cv <- lin_fit$coefficients[1]
        lin_w_cv   <- as.numeric(lin_fit$coefficients[-1])
        lin_w_cv[!is.finite(lin_w_cv)] <- 0
      }
    }

    for (fold in seq_len(nfolds)) {
      val_idx   <- which(foldid == fold)
      train_idx <- which(foldid != fold)
      Z_tr <- lapply(Z_lags, function(x) x[train_idx, , drop = FALSE])
      Z_vl <- lapply(Z_lags, function(x) x[val_idx,   , drop = FALSE])

      fit <- tryCatch({
        train_fa_midas(
          Z_tr, y[train_idx], w[train_idx],
          Z_vl, y[val_idx],   w[val_idx],
          K = K, d = d, L = L_cv, r = params$r,
          H_factor = params$H_factor, n_factor_layers = params$n_factor_layers,
          p_dropout = params$p_dropout,
          W_fixed = W_fixed_cv,
          lin_intercept = lin_int_cv, lin_weights = lin_w_cv,
          lr = params$lr, weight_decay = params$weight_decay,
          n_epochs = 300L, patience = params$patience,
          grad_clip = params$grad_clip
        )
      }, error = function(e) NULL)

      if (is.null(fit)) next
      val_preds <- predict_fa_midas(fit$model, Z_vl, W_fixed_cv)

      if (!is.null(data_for_auc)) {
        auc_val <- tryCatch({
          survivalROC(Stime = data_for_auc$time[val_idx],
                      status = data_for_auc$status[val_idx],
                      marker = val_preds, predict.time = t_horizon,
                      span = 0.25 * length(val_idx)^(-1/2))$AUC
        }, error = function(e) NA)
        if (!is.na(auc_val)) fold_aucs <- c(fold_aucs, auc_val)
      }
    }
    if (length(fold_aucs) >= 2) {
      mean_auc <- mean(fold_aucs)
      if (mean_auc > best_auc) { best_auc <- mean_auc; best_params <- params }
    }
  }
  list(best_params = best_params, best_cv_auc = best_auc)
}

train_ensemble_nam <- function(Z_lags_train, y_train, w_train,
                               Z_lags_test, K, d, jmax, params,
                               use_neural_agg = TRUE,
                               use_residual = FALSE,
                               use_factor = FALSE,
                               W_fixed = NULL,
                               n_restarts = 2,
                               save_weights = FALSE) {
  n_test <- nrow(Z_lags_test[[1]])
  all_preds <- matrix(NA, nrow = n_test, ncol = n_restarts)
  all_gammas <- vector("list", n_restarts)
  all_explained_var <- vector("list", n_restarts)
  all_state_dicts  <- vector("list", n_restarts)
  all_pca_matrices <- vector("list", n_restarts)
  all_arch_specs   <- vector("list", n_restarts)

  r_fac <- if (!is.null(params$r)) params$r else 3L
  H_fac <- if (!is.null(params$H_factor)) params$H_factor else 16L
  n_fac_layers <- if (!is.null(params$n_factor_layers)) params$n_factor_layers else 1L
  patience_ens <- if (!is.null(params$patience)) as.integer(params$patience) else 40L
  grad_clip_ens <- if (!is.null(params$grad_clip)) params$grad_clip else 1.0
  L_ens <- if (!is.null(params$L)) as.integer(params$L) else 3L

  W_fixed_ens <- gb(degree = L_ens - 1L, alpha = -1/2, a = 0, b = 1, jmax = jmax) / jmax

  train_fn <- if (use_factor) train_fanam_midas else train_nam_midas
  extra_args <- if (use_factor) {
    list(r = r_fac, H_factor = H_fac, n_factor_layers = n_fac_layers)
  } else { list() }

  last_model <- NULL
  last_std_test <- NULL
  last_pca_res <- NULL

  for (rs in seq_len(n_restarts)) {
    split <- stratified_split(y_train, val_frac = 0.15)
    Z_tr <- lapply(Z_lags_train, function(x) x[split$train, , drop = FALSE])
    Z_vl <- lapply(Z_lags_train, function(x) x[split$val, , drop = FALSE])
    std_internal <- standardise_lag_groups(Z_tr, Z_vl)
    std_full <- standardise_lag_groups(Z_lags_train, Z_lags_test)

    common_args <- list(
      Z_lags_train = std_internal$train, y_train = y_train[split$train],
      w_train = w_train[split$train],
      Z_lags_val = std_internal$test, y_val = y_train[split$val],
      w_val = w_train[split$val],
      K = K, d = d, L = L_ens,
      H_agg = params$H_agg, n_agg_layers = params$n_agg_layers,
      H_pred = params$H_pred, n_pred_layers = params$n_pred_layers,
      p_dropout = params$p_dropout, p_group_dropout = params$p_group_dropout,
      use_neural_agg = use_neural_agg, use_residual = use_residual,
      W_fixed = W_fixed_ens,
      lr = params$lr, weight_decay = params$weight_decay,
      lambda1 = params$lambda1, lambda2 = params$lambda2,
      output_reg = params$output_reg,
      n_epochs = 500L, patience = patience_ens,
      grad_clip = grad_clip_ens
    )

    fit <- tryCatch({
      do.call(train_fn, c(common_args, extra_args))
    }, error = function(e) NULL)

    if (is.null(fit)) next
    best_n <- max(fit$best_epoch, 20L)

    common_args_full <- common_args
    common_args_full$Z_lags_train <- std_full$train
    common_args_full$y_train <- y_train
    common_args_full$w_train <- w_train
    common_args_full$Z_lags_val <- std_full$train
    common_args_full$y_val <- y_train
    common_args_full$w_val <- w_train
    common_args_full$n_epochs <- as.integer(best_n)
    common_args_full$patience <- as.integer(best_n + 1L)

    fit_final <- tryCatch({
      do.call(train_fn, c(common_args_full, extra_args))
    }, error = function(e) NULL)

    use_fit <- if (!is.null(fit_final)) fit_final else fit
    all_preds[, rs] <- predict_nam_midas(use_fit$model, std_full$test, W_fixed_ens)
    all_gammas[[rs]] <- use_fit$gammas
    if (use_factor && !is.null(use_fit$pca_res)) {
      all_explained_var[[rs]] <- use_fit$pca_res$explained_var
    }
    last_model <- use_fit$model
    last_std_test <- std_full$test
    last_pca_res <- if (use_factor && !is.null(use_fit$pca_res)) use_fit$pca_res else NULL

    if (isTRUE(save_weights)) {
      m <- use_fit$model
      sd_safe <- function(net) {
        if (is.null(net)) return(NULL)
        lapply(net$state_dict(), function(t) as.array(t$cpu()))
      }
      sd_module_list <- function(mlist) {
        if (is.null(mlist)) return(NULL)
        lapply(seq_along(mlist), function(k) sd_safe(mlist[[k]]))
      }
      all_state_dicts[[rs]] <- list(
        bias       = if (!is.null(m$bias)) as.numeric(m$bias$cpu()) else NULL,
        agg_nets   = if (!is.null(m$agg_nets))   sd_module_list(m$agg_nets)   else NULL,
        pred_nets  = if (!is.null(m$pred_nets))  sd_module_list(m$pred_nets)  else NULL,
        factor_net = if (!is.null(m$factor_net)) sd_safe(m$factor_net)        else NULL,
        gamma_raw  = if (!is.null(m$gamma_raw))  as.numeric(m$gamma_raw$cpu()) else NULL
      )
      all_pca_matrices[[rs]] <- if (!is.null(use_fit$pca_res)) use_fit$pca_res$P else NULL
      all_arch_specs[[rs]] <- list(
        d = as.integer(d), K = as.integer(K), L = L_ens,
        r = if (use_factor && !is.null(use_fit$pca_res)) as.integer(use_fit$pca_res$r_used) else NA,
        H_agg = as.integer(params$H_agg),
        n_agg_layers = as.integer(params$n_agg_layers),
        H_pred = as.integer(params$H_pred),
        n_pred_layers = as.integer(params$n_pred_layers),
        H_factor = if (use_factor) as.integer(H_fac) else NA,
        n_factor_layers = if (use_factor) as.integer(n_fac_layers) else NA,
        p_dropout = params$p_dropout,
        p_group_dropout = params$p_group_dropout,
        use_neural_agg = use_neural_agg,
        use_residual = use_residual,
        use_factor = use_factor
      )
    }
  }

  valid_cols <- which(colSums(!is.na(all_preds)) > 0)
  if (length(valid_cols) == 0) return(NULL)

  valid_gammas <- all_gammas[valid_cols]
  avg_gammas <- if (any(!sapply(valid_gammas, is.null))) {
    gamma_mat <- do.call(rbind, Filter(Negate(is.null), valid_gammas))
    colMeans(gamma_mat)
  } else { NULL }

  valid_ev <- Filter(Negate(is.null), all_explained_var[valid_cols])
  avg_ev <- if (length(valid_ev) > 0) {
    colMeans(do.call(rbind, valid_ev))
  } else { NULL }

  diagnostics <- if (!is.null(last_model)) {
    tryCatch({
      extract_contributions(last_model, last_std_test, W_fixed_ens, is_fanam = use_factor)
    }, error = function(e) NULL)
  } else { NULL }

  list(
    predictions = rowMeans(all_preds[, valid_cols, drop = FALSE], na.rm = TRUE),
    n_models = length(valid_cols),
    gammas = avg_gammas,
    explained_var = avg_ev,
    diagnostics = diagnostics,
    pca_matrix = if (!is.null(last_pca_res)) last_pca_res$P else NULL,
    state_dicts  = if (isTRUE(save_weights)) all_state_dicts[valid_cols]  else NULL,
    pca_matrices = if (isTRUE(save_weights)) all_pca_matrices[valid_cols] else NULL,
    arch_specs   = if (isTRUE(save_weights)) all_arch_specs[valid_cols]   else NULL
  )
}

train_ensemble_deep_midas <- function(Z_lags_train, y_train, w_train,
                                      Z_lags_test, K, d, jmax, params,
                                      use_neural_agg = TRUE,
                                      W_fixed = NULL,
                                      n_restarts = 2,
                                      save_weights = FALSE) {
  n_test <- nrow(Z_lags_test[[1]])
  all_preds <- matrix(NA, nrow = n_test, ncol = n_restarts)

  all_state_dicts <- vector("list", n_restarts)
  all_arch_specs  <- vector("list", n_restarts)

  patience_ens <- if (!is.null(params$patience)) as.integer(params$patience) else 40L
  grad_clip_ens <- if (!is.null(params$grad_clip)) params$grad_clip else 1.0
  L_ens <- if (!is.null(params$L)) as.integer(params$L) else 3L

  W_fixed_ens <- gb(degree = L_ens - 1L, alpha = -1/2, a = 0, b = 1, jmax = jmax) / jmax
  W_fixed_for_call <- if (!use_neural_agg) W_fixed_ens else NULL

  for (r in seq_len(n_restarts)) {
    split <- stratified_split(y_train, val_frac = 0.15)
    Z_tr <- lapply(Z_lags_train, function(x) x[split$train, , drop = FALSE])
    Z_vl <- lapply(Z_lags_train, function(x) x[split$val, , drop = FALSE])
    std_internal <- standardise_lag_groups(Z_tr, Z_vl)
    std_full <- standardise_lag_groups(Z_lags_train, Z_lags_test)

    fit <- tryCatch({
      train_deep_midas(
        std_internal$train, y_train[split$train], w_train[split$train],
        std_internal$test, y_train[split$val], w_train[split$val],
        K = K, d = d, L = L_ens,
        H_agg = params$H_agg, H_pred = params$H_pred,
        n_pred_layers = params$n_pred_layers,
        n_agg_layers = params$n_agg_layers,
        p_dropout = params$p_dropout, p_group_dropout = params$p_group_dropout,
        use_neural_agg = use_neural_agg,
        W_fixed = W_fixed_for_call,
        lr = params$lr, weight_decay = params$weight_decay,
        lambda1 = params$lambda1, lambda2 = params$lambda2,
        n_epochs = 500L, patience = patience_ens,
        grad_clip = grad_clip_ens, verbose = FALSE
      )
    }, error = function(e) NULL)

    if (is.null(fit)) next
    best_n <- max(fit$best_epoch, 20L)

    fit_final <- tryCatch({
      train_deep_midas(
        std_full$train, y_train, w_train,
        std_full$train, y_train, w_train,
        K = K, d = d, L = L_ens,
        H_agg = params$H_agg, H_pred = params$H_pred,
        n_pred_layers = params$n_pred_layers,
        n_agg_layers = params$n_agg_layers,
        p_dropout = params$p_dropout, p_group_dropout = params$p_group_dropout,
        use_neural_agg = use_neural_agg,
        W_fixed = W_fixed_for_call,
        lr = params$lr, weight_decay = params$weight_decay,
        lambda1 = params$lambda1, lambda2 = params$lambda2,
        n_epochs = as.integer(best_n), patience = as.integer(best_n + 1L),
        grad_clip = grad_clip_ens, verbose = FALSE
      )
    }, error = function(e) NULL)

    if (!is.null(fit_final)) {
      all_preds[, r] <- predict_deep_midas(fit_final$model, std_full$test, W_fixed_for_call)
      use_fit <- fit_final
    } else {
      all_preds[, r] <- predict_deep_midas(fit$model, std_full$test, W_fixed_for_call)
      use_fit <- fit
    }

    if (isTRUE(save_weights)) {
      m <- use_fit$model
      sd_safe <- function(net) {
        if (is.null(net)) return(NULL)
        lapply(net$state_dict(), function(t) as.array(t$cpu()))
      }
      all_state_dicts[[r]] <- list(
        agg_net = if (use_neural_agg) sd_safe(m$agg_net) else NULL,
        pred    = sd_safe(m$pred)
      )
      all_arch_specs[[r]] <- list(
        d = d, K = K, L = L_ens,
        H_agg = params$H_agg, H_pred = params$H_pred,
        n_pred_layers = params$n_pred_layers,
        n_agg_layers = params$n_agg_layers,
        p_dropout = params$p_dropout,
        p_group_dropout = params$p_group_dropout,
        use_neural_agg = use_neural_agg
      )
    }
  }

  valid_cols <- which(colSums(!is.na(all_preds)) > 0)
  if (length(valid_cols) == 0) return(NULL)

  list(
    predictions = rowMeans(all_preds[, valid_cols, drop = FALSE], na.rm = TRUE),
    n_models = length(valid_cols),
    state_dicts = if (isTRUE(save_weights)) all_state_dicts[valid_cols] else NULL,
    arch_specs  = if (isTRUE(save_weights)) all_arch_specs[valid_cols]  else NULL
  )
}
train_ensemble_fa_midas <- function(Z_lags_train, y_train, w_train,
                                    Z_lags_test, K, d, jmax, params,
                                    W_fixed = NULL,
                                    lin_intercept, lin_weights,
                                    n_restarts = 2) {
  n_test <- nrow(Z_lags_test[[1]])
  all_preds <- matrix(NA, nrow = n_test, ncol = n_restarts)
  all_explained_var <- vector("list", n_restarts)

  patience_ens   <- if (!is.null(params$patience))  as.integer(params$patience) else 40L
  grad_clip_ens  <- if (!is.null(params$grad_clip)) params$grad_clip else 1.0
  L_ens          <- if (!is.null(params$L))         as.integer(params$L) else 3L
  r_fac          <- if (!is.null(params$r))         as.integer(params$r) else 3L
  H_fac          <- if (!is.null(params$H_factor))  as.integer(params$H_factor) else 16L
  n_fac_layers   <- if (!is.null(params$n_factor_layers)) as.integer(params$n_factor_layers) else 1L

  W_fixed_ens <- gb(degree = L_ens - 1L, alpha = -1/2, a = 0, b = 1, jmax = jmax) / jmax

  if (length(lin_weights) != K * L_ens) {
    H_full <- matrix(0, nrow = length(y_train), ncol = K * L_ens)
    for (k in seq_len(K)) {
      H_full[, ((k - 1L) * L_ens + 1L):(k * L_ens)] <-
        as.matrix(Z_lags_train[[k]]) %*% as.matrix(W_fixed_ens)
    }
    lin_fit <- tryCatch({
      suppressWarnings(glm.fit(x = cbind(1, H_full), y = y_train, weights = w_train,
                               family = binomial()))
    }, error = function(e) NULL)
    if (!is.null(lin_fit) && all(is.finite(lin_fit$coefficients))) {
      lin_intercept <- lin_fit$coefficients[1]
      lin_weights   <- as.numeric(lin_fit$coefficients[-1])
      lin_weights[!is.finite(lin_weights)] <- 0
    } else {
      lin_intercept <- 0
      lin_weights   <- rep(0, K * L_ens)
    }
  }

  for (rs in seq_len(n_restarts)) {
    split <- stratified_split(y_train, val_frac = 0.15)
    Z_tr <- lapply(Z_lags_train, function(x) x[split$train, , drop = FALSE])
    Z_vl <- lapply(Z_lags_train, function(x) x[split$val,   , drop = FALSE])

    fit <- tryCatch({
      train_fa_midas(
        Z_tr, y_train[split$train], w_train[split$train],
        Z_vl, y_train[split$val],   w_train[split$val],
        K = K, d = d, L = L_ens, r = r_fac,
        H_factor = H_fac, n_factor_layers = n_fac_layers,
        p_dropout = params$p_dropout,
        W_fixed = W_fixed_ens,
        lin_intercept = lin_intercept, lin_weights = lin_weights,
        lr = params$lr, weight_decay = params$weight_decay,
        n_epochs = 500L, patience = patience_ens,
        grad_clip = grad_clip_ens
      )
    }, error = function(e) NULL)

    if (is.null(fit)) next
    best_n <- max(fit$best_epoch, 20L)

    fit_final <- tryCatch({
      train_fa_midas(
        Z_lags_train, y_train, w_train,
        Z_lags_train, y_train, w_train,
        K = K, d = d, L = L_ens, r = r_fac,
        H_factor = H_fac, n_factor_layers = n_fac_layers,
        p_dropout = params$p_dropout,
        W_fixed = W_fixed_ens,
        lin_intercept = lin_intercept, lin_weights = lin_weights,
        lr = params$lr, weight_decay = params$weight_decay,
        n_epochs = as.integer(best_n), patience = as.integer(best_n + 1L),
        grad_clip = grad_clip_ens
      )
    }, error = function(e) NULL)

    use_fit <- if (!is.null(fit_final)) fit_final else fit
    all_preds[, rs] <- predict_fa_midas(use_fit$model, Z_lags_test, W_fixed_ens)
    if (!is.null(use_fit$pca_res)) {
      all_explained_var[[rs]] <- use_fit$pca_res$explained_var
    }
  }

  valid_cols <- which(colSums(!is.na(all_preds)) > 0)
  if (length(valid_cols) == 0) return(NULL)

  valid_ev <- Filter(Negate(is.null), all_explained_var[valid_cols])
  avg_ev <- if (length(valid_ev) > 0) {
    colMeans(do.call(rbind, valid_ev))
  } else { NULL }

  list(
    predictions = rowMeans(all_preds[, valid_cols, drop = FALSE], na.rm = TRUE),
    n_models = length(valid_cols),
    explained_var = avg_ev
  )
}

run_one_iteration <- function(iter, data, s, t_horizon, jmax, num_vars, degree,
                              w_fin, alpha, intercept_zero, bootstrap_number,
                              n_random_cv) {

  torch_set_num_threads(1L)
  Sys.setenv(OPENBLAS_NUM_THREADS = "1")
  Sys.setenv(MKL_NUM_THREADS      = "1")
  Sys.setenv(OMP_NUM_THREADS      = "1")
  Sys.setenv(BLIS_NUM_THREADS     = "1")

  log_file <- "baseline_progress.log"
  log_msg <- function(msg) {
    line <- sprintf("[%s] t=%.1f iter=%02d: %s\n", format(Sys.time(), "%H:%M:%S"), t_horizon, iter, msg)
    tryCatch(cat(line, file = log_file, append = TRUE), error = function(e) NULL)
  }
  log_msg("START")

  single_agg_net <<- nn_module(
    "SingleAggNet",
    initialize = function(d_in, L, hidden_size = 16L, n_layers = 1L) {
      self$n_layers <- as.integer(n_layers)
      self$act <- nn_elu()
      self$fc1 <- nn_linear(d_in, hidden_size)
      floor_w <- as.integer(max(4L, as.integer(L)))
      if (self$n_layers >= 2L) {
        h2 <- as.integer(max(floor(hidden_size / 2), floor_w))
        self$fc2 <- nn_linear(hidden_size, h2)
        if (self$n_layers >= 3L) {
          h3 <- as.integer(max(floor(h2 / 2), floor_w))
          self$fc3 <- nn_linear(h2, h3)
          if (self$n_layers >= 4L) {
            h4 <- as.integer(max(floor(h3 / 2), floor_w))
            self$fc4 <- nn_linear(h3, h4)
            self$fc_out <- nn_linear(h4, L)
          } else { self$fc_out <- nn_linear(h3, L) }
        } else { self$fc_out <- nn_linear(h2, L) }
      } else { self$fc_out <- nn_linear(hidden_size, L) }
    },
    forward = function(x) {
      x <- self$act(self$fc1(x))
      if (self$n_layers >= 2L) {
        x <- self$act(self$fc2(x))
        if (self$n_layers >= 3L) {
          x <- self$act(self$fc3(x))
          if (self$n_layers >= 4L) x <- self$act(self$fc4(x))
        }
      }
      self$fc_out(x)
    }
  )

  single_pred_net <<- nn_module(
    "SinglePredNet",
    initialize = function(L_in, hidden_size = 8L, n_layers = 1L, p_dropout = 0.2) {
      self$n_layers <- as.integer(n_layers)
      self$act <- nn_relu()
      self$drop <- nn_dropout(p = p_dropout)
      self$fc1 <- nn_linear(L_in, hidden_size)
      if (self$n_layers >= 2L) {
        h2 <- as.integer(max(floor(hidden_size / 2), 4L))
        self$fc2 <- nn_linear(hidden_size, h2)
        if (self$n_layers >= 3L) {
          h3 <- as.integer(max(floor(h2 / 2), 4L))
          self$fc3 <- nn_linear(h2, h3)
          if (self$n_layers >= 4L) {
            h4 <- as.integer(max(floor(h3 / 2), 4L))
            self$fc4 <- nn_linear(h3, h4)
            self$fc_out <- nn_linear(h4, 1L)
          } else { self$fc_out <- nn_linear(h3, 1L) }
        } else { self$fc_out <- nn_linear(h2, 1L) }
      } else { self$fc_out <- nn_linear(hidden_size, 1L) }
    },
    forward = function(x) {
      x <- self$drop(self$act(self$fc1(x)))
      if (self$n_layers >= 2L) {
        x <- self$drop(self$act(self$fc2(x)))
        if (self$n_layers >= 3L) {
          x <- self$drop(self$act(self$fc3(x)))
          if (self$n_layers >= 4L) x <- self$drop(self$act(self$fc4(x)))
        }
      }
      self$fc_out(x)
    }
  )

  build_joint_pred <<- function(input_dim, H_pred, n_pred_layers, p_dropout) {
    input_dim <- as.integer(input_dim)
    H_pred <- as.integer(H_pred)
    if (n_pred_layers == 1L) {
      nn_sequential(
        nn_linear(input_dim, H_pred), nn_relu(), nn_dropout(p = p_dropout),
        nn_linear(H_pred, 1L))
    } else if (n_pred_layers == 2L) {
      H2 <- as.integer(max(floor(H_pred / 2), 8L))
      nn_sequential(
        nn_linear(input_dim, H_pred), nn_relu(), nn_dropout(p = p_dropout),
        nn_linear(H_pred, H2), nn_relu(), nn_dropout(p = p_dropout),
        nn_linear(H2, 1L))
    } else if (n_pred_layers == 3L) {
      H2 <- as.integer(max(floor(H_pred / 2), 8L))
      H3 <- as.integer(max(floor(H_pred / 4), 4L))
      nn_sequential(
        nn_linear(input_dim, H_pred), nn_relu(), nn_dropout(p = p_dropout),
        nn_linear(H_pred, H2), nn_relu(), nn_dropout(p = p_dropout),
        nn_linear(H2, H3), nn_relu(), nn_dropout(p = p_dropout),
        nn_linear(H3, 1L))
    } else {
      H2 <- as.integer(max(floor(H_pred / 2), 8L))
      H3 <- as.integer(max(floor(H_pred / 4), 4L))
      H4 <- as.integer(max(floor(H_pred / 8), 4L))
      nn_sequential(
        nn_linear(input_dim, H_pred), nn_relu(), nn_dropout(p = p_dropout),
        nn_linear(H_pred, H2), nn_relu(), nn_dropout(p = p_dropout),
        nn_linear(H2, H3), nn_relu(), nn_dropout(p = p_dropout),
        nn_linear(H3, H4), nn_relu(), nn_dropout(p = p_dropout),
        nn_linear(H4, 1L))
    }
  }

  deep_midas_module <<- nn_module(
    "DeepMIDAS",
    initialize = function(d, K, L = 3L, H_agg = 16L, H_pred = 32L,
                          n_pred_layers = 1L, n_agg_layers = 1L,
                          p_dropout = 0.2, p_group_dropout = 0.1,
                          use_neural_agg = TRUE) {
      self$K <- as.integer(K); self$L <- as.integer(L)
      self$p_group_dropout <- p_group_dropout
      self$use_neural_agg <- use_neural_agg
      if (use_neural_agg) {
        self$agg_net <- single_agg_net(as.integer(d), self$L, as.integer(H_agg),
                                       n_layers = as.integer(n_agg_layers))
      }
      input_dim <- as.integer(self$K * self$L + 1L)
      self$pred <- build_joint_pred(input_dim, H_pred, n_pred_layers, p_dropout)
    },
    forward = function(Z_lags, W_fixed = NULL) {
      batch_size <- Z_lags[[1]]$shape[1]
      h_list <- vector("list", self$K)
      for (k in seq_len(self$K)) {
        if (self$use_neural_agg) { h_list[[k]] <- self$agg_net(Z_lags[[k]])
        } else { h_list[[k]] <- torch_mm(Z_lags[[k]], W_fixed) }
      }
      h <- torch_stack(h_list, dim = 2)
      if (self$training && self$p_group_dropout > 0) {
        mask <- (torch_rand(batch_size, self$K, 1L) > self$p_group_dropout)
        mask <- mask$to(dtype = h$dtype, device = h$device)
        h <- h * mask
      }
      h_flat <- h$reshape(c(batch_size, -1))
      intercept <- torch_ones(batch_size, 1L, dtype = h$dtype, device = h$device)
      h_input <- torch_cat(list(intercept, h_flat), dim = 2)
      self$pred(h_input)$squeeze(dim = 2)
    }
  )

  compute_group_penalty <<- function(model, K, L, lambda1 = 0.01, lambda2 = 0.01) {
    W_first <- model$pred[[1]]$weight
    W_groups <- W_first[, 2:(K * L + 1)]
    l1_pen <- torch_tensor(0, dtype = W_groups$dtype, device = W_groups$device, requires_grad = FALSE)
    l2_pen <- torch_tensor(0, dtype = W_groups$dtype, device = W_groups$device, requires_grad = FALSE)
    for (k in seq_len(K)) {
      col_start <- (k - 1L) * L + 1L
      col_end <- k * L
      W_k <- W_groups[, col_start:col_end]
      l1_pen <- l1_pen + W_k$abs()$sum()
      l2_pen <- l2_pen + W_k$norm(2)
    }
    lambda1 * l1_pen + lambda2 * l2_pen
  }

  nam_midas_module <<- nn_module(
    "NAM_MIDAS",
    initialize = function(d, K, L = 3L,
                          H_agg = 16L, n_agg_layers = 1L,
                          H_pred = 8L, n_pred_layers = 1L,
                          p_dropout = 0.2, p_group_dropout = 0.1,
                          use_neural_agg = TRUE, use_residual = FALSE) {
      self$K <- as.integer(K); self$L <- as.integer(L)
      self$p_group_dropout <- p_group_dropout
      self$use_neural_agg <- use_neural_agg
      self$use_residual <- use_residual
      if (use_neural_agg || use_residual) {
        self$agg_nets <- nn_module_list(
          lapply(seq_len(K), function(k) {
            single_agg_net(as.integer(d), as.integer(L),
                           hidden_size = as.integer(H_agg),
                           n_layers = as.integer(n_agg_layers))
          })
        )
      }
      if (use_residual) self$gamma_raw <- nn_parameter(torch_full(K, -2.0))
      self$pred_nets <- nn_module_list(
        lapply(seq_len(K), function(k) {
          single_pred_net(as.integer(L),
                          hidden_size = as.integer(H_pred),
                          n_layers = as.integer(n_pred_layers),
                          p_dropout = p_dropout)
        })
      )
      self$bias <- nn_parameter(torch_zeros(1L))
    },
    forward = function(Z_lags, W_fixed = NULL) {
      gammas <- if (self$use_residual) torch_sigmoid(self$gamma_raw) else NULL
      contrib_list <- vector("list", self$K)
      for (k in seq_len(self$K)) {
        if (self$use_residual) {
          h_fixed <- torch_mm(Z_lags[[k]], W_fixed)
          h_neural <- self$agg_nets[[k]](Z_lags[[k]])
          h_k <- (1 - gammas[k]) * h_fixed + gammas[k] * h_neural
        } else if (self$use_neural_agg) {
          h_k <- self$agg_nets[[k]](Z_lags[[k]])
        } else {
          h_k <- torch_mm(Z_lags[[k]], W_fixed)
        }
        contrib_list[[k]] <- self$pred_nets[[k]](h_k)$squeeze(dim = 2)
      }
      stacked <- torch_stack(contrib_list, dim = 2)
      if (self$training && self$p_group_dropout > 0) {
        stacked <- nn_dropout(p = self$p_group_dropout)(stacked)
      }
      stacked$sum(dim = 2) + self$bias
    },
    get_gammas = function() {
      if (!is.null(self$gamma_raw)) {
        as.numeric(torch_sigmoid(self$gamma_raw)$cpu())
      } else { NULL }
    }
  )

  fanam_midas_module <<- nn_module(
    "FANAM_MIDAS",
    initialize = function(d, K, L = 3L, r = 3L,
                          H_agg = 16L, n_agg_layers = 1L,
                          H_pred = 8L, n_pred_layers = 1L,
                          H_factor = 16L, n_factor_layers = 1L,
                          p_dropout = 0.2, p_group_dropout = 0.1,
                          use_neural_agg = TRUE, use_residual = FALSE) {
      self$K <- as.integer(K); self$L <- as.integer(L); self$r <- as.integer(r)
      self$p_group_dropout <- p_group_dropout
      self$use_neural_agg <- use_neural_agg
      self$use_residual <- use_residual
      if (use_neural_agg || use_residual) {
        self$agg_nets <- nn_module_list(
          lapply(seq_len(K), function(k) {
            single_agg_net(as.integer(d), as.integer(L),
                           hidden_size = as.integer(H_agg),
                           n_layers = as.integer(n_agg_layers))
          })
        )
      }
      if (use_residual) self$gamma_raw <- nn_parameter(torch_full(K, -2.0))
      self$factor_proj <- NULL
      H_factor <- as.integer(H_factor)
      if (as.integer(n_factor_layers) >= 2L) {
        h2f <- as.integer(max(floor(H_factor / 2), 4L))
        self$factor_net <- nn_sequential(
          nn_linear(as.integer(r), H_factor), nn_relu(), nn_dropout(p = p_dropout),
          nn_linear(H_factor, h2f), nn_relu(), nn_dropout(p = p_dropout),
          nn_linear(h2f, 1L))
      } else {
        self$factor_net <- nn_sequential(
          nn_linear(as.integer(r), H_factor), nn_relu(), nn_dropout(p = p_dropout),
          nn_linear(H_factor, 1L))
      }
      self$pred_nets <- nn_module_list(
        lapply(seq_len(K), function(k) {
          single_pred_net(as.integer(L),
                          hidden_size = as.integer(H_pred),
                          n_layers = as.integer(n_pred_layers),
                          p_dropout = p_dropout)
        })
      )
      self$bias <- nn_parameter(torch_zeros(1L))
    },
    set_factor_projection = function(P_matrix) {
      self$factor_proj <- torch_tensor(as.matrix(P_matrix), dtype = torch_float())
    },
    forward = function(Z_lags, W_fixed = NULL) {
      gammas <- if (self$use_residual) torch_sigmoid(self$gamma_raw) else NULL
      h_list <- vector("list", self$K)
      contrib_list <- vector("list", self$K)
      for (k in seq_len(self$K)) {
        if (self$use_residual) {
          h_fixed <- torch_mm(Z_lags[[k]], W_fixed)
          h_neural <- self$agg_nets[[k]](Z_lags[[k]])
          h_k <- (1 - gammas[k]) * h_fixed + gammas[k] * h_neural
        } else if (self$use_neural_agg) {
          h_k <- self$agg_nets[[k]](Z_lags[[k]])
        } else {
          h_k <- torch_mm(Z_lags[[k]], W_fixed)
        }
        h_list[[k]] <- h_k
        contrib_list[[k]] <- self$pred_nets[[k]](h_k)$squeeze(dim = 2)
      }
      h_stacked <- torch_cat(h_list, dim = 2)
      f_tilde <- torch_mm(h_stacked, self$factor_proj$to(device = h_stacked$device))
      factor_contrib <- self$factor_net(f_tilde)$squeeze(dim = 2)
      stacked <- torch_stack(contrib_list, dim = 2)
      if (self$training && self$p_group_dropout > 0) {
        stacked <- nn_dropout(p = self$p_group_dropout)(stacked)
      }
      factor_contrib + stacked$sum(dim = 2) + self$bias
    },
    get_gammas = function() {
      if (!is.null(self$gamma_raw)) {
        as.numeric(torch_sigmoid(self$gamma_raw)$cpu())
      } else { NULL }
    }
  )

  fa_midas_module <<- nn_module(
    "FA_MIDAS",
    initialize = function(d, K, L = 3L, r = 3L,
                          H_factor = 16L, n_factor_layers = 1L,
                          p_dropout = 0.2) {
      self$K <- as.integer(K); self$L <- as.integer(L); self$r <- as.integer(r)
      self$factor_proj   <- NULL
      self$lin_intercept <- NULL
      self$lin_weights   <- NULL
      H_factor <- as.integer(H_factor)
      if (as.integer(n_factor_layers) >= 2L) {
        h2f <- as.integer(max(floor(H_factor / 2), 4L))
        self$factor_net <- nn_sequential(
          nn_linear(as.integer(r), H_factor), nn_relu(), nn_dropout(p = p_dropout),
          nn_linear(H_factor, h2f), nn_relu(), nn_dropout(p = p_dropout),
          nn_linear(h2f, 1L))
      } else {
        self$factor_net <- nn_sequential(
          nn_linear(as.integer(r), H_factor), nn_relu(), nn_dropout(p = p_dropout),
          nn_linear(H_factor, 1L))
      }
    },
    set_factor_projection = function(P_matrix) {
      self$factor_proj <- torch_tensor(as.matrix(P_matrix), dtype = torch_float())
    },
    set_linear_offset = function(intercept, weights) {
      self$lin_intercept <- torch_tensor(as.numeric(intercept), dtype = torch_float())
      self$lin_weights   <- torch_tensor(as.numeric(weights),   dtype = torch_float())
    },
    forward = function(Z_lags, W_fixed) {
      h_list <- vector("list", self$K)
      for (k in seq_len(self$K)) h_list[[k]] <- torch_mm(Z_lags[[k]], W_fixed)
      h_stacked <- torch_cat(h_list, dim = 2)
      lin_w <- self$lin_weights$to(device = h_stacked$device)
      lin_b <- self$lin_intercept$to(device = h_stacked$device)
      base_logit <- torch_mv(h_stacked, lin_w) + lin_b
      f_tilde <- torch_mm(h_stacked, self$factor_proj$to(device = h_stacked$device))
      factor_contrib <- self$factor_net(f_tilde)$squeeze(dim = 2)
      base_logit + factor_contrib
    }
  )

  set.seed(s * iter)
  torch_manual_seed(s * iter)

  dataset_p <- data[which((1 * (data$time <= t_horizon) * 1 * (data$status == 1)) == 1), ]
  dataset_n <- data[which((1 * (data$time <= t_horizon) * 1 * (data$status == 1)) == 0), ]

  if (nrow(dataset_p) < 5) return(NULL)

  indices_p <- sample(nrow(dataset_p), nrow(dataset_p) * 0.8)
  indices_n <- sample(nrow(dataset_n), nrow(dataset_n) * 0.8)
  train_dataset_in <- rbind(dataset_p[indices_p, ], dataset_n[indices_n, ])
  test_dataset_in  <- rbind(dataset_p[-indices_p, ], dataset_n[-indices_n, ])

  train_dataset <- KM_estimate(testRandomLogitDataset(train_dataset_in, t = t_horizon))
  test_dataset  <- KM_estimate(testRandomLogitDataset(test_dataset_in, t = t_horizon))

  num_cols_to_use <- num_vars * jmax
  X_train <- apply(train_dataset[, 1:num_cols_to_use], 2, as.numeric)
  X_test  <- apply(test_dataset[, 1:num_cols_to_use], 2, as.numeric)
  y_train <- as.numeric(train_dataset$time <= t_horizon)
  y_test  <- as.numeric(test_dataset$time <= t_horizon)

  w_train <- (1 - 1 * (train_dataset$time <= t_horizon) * (1 - train_dataset$status)) / train_dataset$Gp
  w_train[is.na(w_train) | is.infinite(w_train)] <- 0
  w_test <- (1 - 1 * (test_dataset$time <= t_horizon) * (1 - test_dataset$status)) / test_dataset$Gp
  w_test[is.na(w_test) | is.infinite(w_test)] <- 0

  gindex <- NULL; Xdw <- NULL
  for (z in seq(num_vars)) {
    z_idx <- (1 + (z - 1) * jmax):(z * jmax)
    Xdw <- cbind(Xdw, X_train[, z_idx] %*% w_fin)
    gindex <- c(gindex, rep(z, times = degree + 1))
  }
  X_in <- as.matrix(cbind(rep(1, nrow(X_train)), Xdw))
  foldid_bl <- form_folds(nrow(X_in), 5)

  fit_cv_MIDAS <- tryCatch({
    alpha_cv_sparsegl(X_in[, -1], y_train, group = gindex, nlambda = 200,
                      weight = w_train, alpha = alpha, nfolds = 5, foldid = foldid_bl,
                      pred.loss = 'censor', intercept_zero = intercept_zero,
                      standardize = TRUE, AUC = TRUE, data = train_dataset, t = t_horizon)
  }, error = function(e) NULL)

  if (is.null(fit_cv_MIDAS) || is.null(fit_cv_MIDAS$coff)) return(NULL)

  est_MIDAS_AUC <- fit_cv_MIDAS$coff_AUC
  X_t <- NULL
  for (z in seq(num_vars)) {
    z_idx <- (1 + (z - 1) * jmax):(z * jmax)
    X_t <- cbind(X_t, X_test[, z_idx] %*% w_fin)
  }
  X_te <- as.matrix(cbind(rep(1, nrow(X_test)), X_t))

  test_preds_bl <- as.numeric(plogis(X_te %*% est_MIDAS_AUC))
  AUC_bl <- ROC_censor_N(data = test_dataset, prediction = test_preds_bl, t = t_horizon)$AUC
  AUC_bl_boot <- ROC_N_bootstrap(data = test_dataset, prediction = test_preds_bl,
                                 t = t_horizon, sim_number = bootstrap_number)
  n_nonzero_bl <- sum(est_MIDAS_AUC[-1] != 0)
  log_msg(sprintf("Baseline done: AUC=%.3f, non-zero=%d", AUC_bl, n_nonzero_bl))

  log_msg(sprintf("DONE: BL=%.3f", AUC_bl))

  full_result <- list(
    AUC_baseline = round(AUC_bl, 3), AUC_baseline_boot = AUC_bl_boot,
    n_nonzero_baseline = n_nonzero_bl,
    est_MIDAS_AUC = est_MIDAS_AUC,
    iter = iter,
    t_horizon = t_horizon
  )

  worker_file <- sprintf("baseline_worker_s%d_t%s_iter%02d.rds", s, t_horizon, iter)
  tryCatch(saveRDS(full_result, worker_file), error = function(e) {
    log_msg(sprintf("WARNING: Failed to save worker file: %s", e$message))
  })
  log_msg(sprintf("Saved worker file: %s", worker_file))

  list(AUC_baseline = round(AUC_bl, 3), iter = iter, saved = TRUE)
}

s <- 6
lag_use_year <- 6
quarter <- 4
lags <- lag_use_year * quarter - 1
jmax <- lags

DATA_FILE <- if (exists("DATA_FILE")) DATA_FILE else "period_final_us_nonfin.csv"
cat(sprintf("Reading data from: %s\n", DATA_FILE))
data_financial <- read.csv2(DATA_FILE)
n <- dim(data_financial)[1]
p_fin <- dim(data_financial)[2] - 5
for (col in names(data_financial)[1:p_fin]) {
  if (grepl("%", col)) data_financial[[col]] <- data_financial[[col]] * 0.01
}
data <- data_financial
p <- dim(data)[2] - 5; n <- dim(data)[1]

end_observation <- '2020-12-31'
data$start_day <- sapply(data$start_day, convert_to_ymd)
data$censoringtime <- as.numeric(as.Date(end_observation) - as.Date(data$start_day)) / 365
data$Ts <- data$survival_time
data$time <- pmin(data$Ts, data$censoringtime)
data$status <- data$status

intercept_zero <- 0
degree <- 2
w_fin <- gb(degree = degree, alpha = -1/2, a = 0, b = 1, jmax = jmax) / jmax
alpha <- c(0, 0.1, 0.3, 0.5, 0.7, 0.9, 1)
bootstrap_number <- 1000

num_vars <- floor(p / jmax)
n_random_cv <- 50L

cat(sprintf("Data: n=%d, K=%d, jmax=%d\n", n, num_vars, jmax))
cat(sprintf("Event rate: %.1f%% (%d events)\n",
            100*mean(data$status), sum(data$status)))
cat(sprintf("Model: sg-LASSO-MIDAS baseline\n"))
cat(sprintf("CV: alpha_cv_sparsegl with 200 lambdas x 5 folds x %d alphas\n", length(alpha)))

num_cores <- detectCores()
n_workers <- min(num_cores, 30L)
cat(sprintf("Using %d parallel workers (of %d cores)\n", n_workers, num_cores))

cl <- makeCluster(n_workers)
registerDoParallel(cl)

worker_packages <- c('torch', 'sparsegl', 'Survivalml', 'midasml', 'dplyr',
                     'survival', 'survivalROC', 'MLmetrics', 'pROC', 'PRROC',
                     'dotCall64', 'glmnet', 'rlang', 'pracma', 'timeROC',
                     'RSpectra')

worker_exports <- c('run_one_iteration',
                    'nam_midas_module', 'fanam_midas_module', 'fa_midas_module',
                    'single_agg_net', 'single_pred_net',
                    'deep_midas_module', 'build_joint_pred',
                    'compute_group_penalty',
                    'ipcw_bce_loss', 'compute_nam_penalty', 'compute_output_penalty',
                    'compute_factor_projection',
                    'extract_contributions',
                    'standardise_lag_groups', 'stratified_split',
                    'train_nam_midas', 'train_fanam_midas',
                    'train_fa_midas', 'predict_fa_midas',
                    'cv_fa_midas', 'train_ensemble_fa_midas',
                    'train_deep_midas', 'predict_deep_midas',
                    'cv_deep_midas', 'train_ensemble_deep_midas',
                    'predict_nam_midas',
                    'extract_lag_groups', 'cv_nam_midas',
                    'train_ensemble_nam',
                    'KM_estimate', 'testRandomLogitDataset',
                    'form_folds', 'alpha_cv_sparsegl',
                    'ROC_censor_N', 'ROC_N_bootstrap', 'gb',
                    'data', 's', 'jmax', 'num_vars', 'degree',
                    'w_fin', 'alpha', 'intercept_zero', 'bootstrap_number',
                    'convert_to_ymd', 'n_random_cv')

it <- 10
all_tasks <- expand.grid(iter = seq_len(it),
                         t_horizon = c(8, 8.5, 9),
                         KEEP.OUT.ATTRS = FALSE,
                         stringsAsFactors = FALSE)
cat(sprintf("Dispatching %d tasks to %d workers\n",
            nrow(all_tasks), n_workers))

start_time <- Sys.time()

result_para <- foreach(
  task_idx = seq_len(nrow(all_tasks)),
  .errorhandling = 'pass',
  .packages = worker_packages,
  .export = worker_exports
) %dopar% {
  iter      <- all_tasks$iter[task_idx]
  t_horizon <- all_tasks$t_horizon[task_idx]
  run_one_iteration(iter, data, s, t_horizon, jmax, num_vars, degree,
                    w_fin, alpha, intercept_zero, bootstrap_number,
                    n_random_cv)
}

elapsed <- difftime(Sys.time(), start_time, units = "hours")
cat(sprintf("\n\nAll %d baseline tasks finished after %.2f hours wall-clock\n\n",
            nrow(all_tasks), as.numeric(elapsed)))

saveRDS(result_para, sprintf("baseline_raw_all_s%d.rds", s))

stopCluster(cl)

for (t_horizon in c(8, 8.5, 9)) {

  cat(sprintf("\n# AGGREGATE BASELINE t = %s\n", t_horizon))

  valid_results <- list()
  for (i in seq_len(it)) {
    wf <- sprintf("baseline_worker_s%d_t%s_iter%02d.rds", s, t_horizon, i)
    if (file.exists(wf)) {
      r <- tryCatch(readRDS(wf), error = function(e) NULL)
      if (!is.null(r) && !is.null(r$AUC_baseline)) {
        valid_results[[length(valid_results) + 1]] <- r
      }
    }
  }

  if (length(valid_results) == 0) {
    cat(sprintf("WARNING: No results for t = %s\n", t_horizon)); next
  }
  cat(sprintf("%d/%d iterations successful\n", length(valid_results), it))

  auc_bl <- sapply(valid_results, function(x) x$AUC_baseline)
  n_nz   <- sapply(valid_results, function(x) x$n_nonzero_baseline)

  cat(sprintf("  Mean baseline AUC = %.3f (sd %.3f)\n",
              mean(auc_bl, na.rm = TRUE), sd(auc_bl, na.rm = TRUE)))
  cat(sprintf("  Baseline non-zero: mean=%.1f, range=[%d, %d] (of 97)\n",
              mean(n_nz), min(n_nz), max(n_nz)))
  for (i in seq_along(valid_results)) {
    r <- valid_results[[i]]
    cat(sprintf("  iter %2d: BL=%.3f, non-zero=%d\n", r$iter, r$AUC_baseline, r$n_nonzero_baseline))
  }
}

cat("\n\n=== Baseline run complete ===\n")
cat(sprintf("Total wall-clock: %.2f hours\n", as.numeric(elapsed)))
cat("Next: run train_nam_family.R for the neural variants.\n")
