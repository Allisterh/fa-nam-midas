# Out-of-sample distress probabilities for the case-study firms.
#
# Produces truly out-of-sample distress probabilities for the six
# case-study firms (AAPL, MSFT, ENRNQ, KODK, CCTYQ, GM) using the
# FA-NAM-MIDAS restarts trained on the panel with those firms held out.
#
# For each (iteration, horizon) pair the script recreates that
# iteration's training split, recomputes the per-(variable, lag)
# standardisation statistics on the training firms, applies them to the
# held-out firms, rebuilds the FA-NAM-MIDAS model from the saved
# architecture specs and state dicts, runs a forward pass, and stores the
# sigmoid probability. Probabilities are then averaged across restarts
# and iterations per firm per horizon.
#
# Inputs
#   results_dir/case_study_panel.rds            written by train_nam_holdout.R
#   results_dir/period_final_us_nonfin.csv      the raw US panel
#   results_dir/period_final_us_nonfin_lookup.csv
#   results_dir/<holdout worker files>          one per (horizon, iteration)
#   results_dir/import functions for empirical application.R
# Outputs
#   case_study_predictions.csv
#   case_study_predictions_full.rds

suppressPackageStartupMessages({
  library(torch)
  library(Survivalml)
  library(survival)
  library(midasml)
})

RESULTS_DIR <- if (exists("RESULTS_DIR")) RESULTS_DIR else getwd()
DATA_FILE   <- file.path(RESULTS_DIR, "period_final_us_nonfin.csv")
CASE_PANEL  <- file.path(RESULTS_DIR, "case_study_panel.rds")
s           <- 6L
ths         <- c(8, 8.5, 9)
its         <- 1:10
jmax        <- 23L
num_vars    <- 32L

stopifnot(file.exists(CASE_PANEL))
stopifnot(file.exists(DATA_FILE))

source(file.path(RESULTS_DIR, "import functions for empirical application.R"))

extract_lag_groups <- function(X_matrix, num_vars, jmax) {
  out <- vector("list", num_vars)
  for (k in seq_len(num_vars)) {
    cs <- (k - 1L) * jmax + 1L; ce <- k * jmax
    out[[k]] <- X_matrix[, cs:ce, drop = FALSE]
  }
  out
}

standardise_with_stats <- function(Z, means_list, sds_list) {
  out <- vector("list", length(Z))
  for (k in seq_along(Z)) {
    out[[k]] <- sweep(sweep(Z[[k]], 2, means_list[[k]], "-"),
                      2, sds_list[[k]], "/")
  }
  out
}

cs <- readRDS(CASE_PANEL)
cs_tickers <- cs$lookup$tic
cs_firms   <- cs$lookup$conm
cs_status  <- cs$lookup$status
cs_bk      <- cs$lookup$bankrupt_day
cs_X       <- as.matrix(sapply(cs$covariates[, 1:(num_vars * jmax)], as.numeric))
cs_X       <- matrix(cs_X, nrow = nrow(cs$covariates), ncol = num_vars * jmax)
n_cs <- nrow(cs_X)
cat(sprintf("Loaded case_study_panel.rds: %d firms\n", n_cs))
for (i in seq_len(n_cs)) {
  cat(sprintf("  %2d. %-8s %s (status=%d, event=%s)\n",
              i, cs_tickers[i], cs_firms[i], cs_status[i], cs_bk[i]))
}

single_agg_net <- nn_module(
  "SingleAggNet",
  initialize = function(d_in, L, hidden_size = 16L, n_layers = 1L) {
    self$n_layers <- as.integer(n_layers); self$act <- nn_elu()
    self$fc1 <- nn_linear(d_in, hidden_size)
    floor_w <- as.integer(max(4L, as.integer(L)))
    if (self$n_layers >= 2L) {
      h2 <- as.integer(max(floor(hidden_size/2), floor_w))
      self$fc2 <- nn_linear(hidden_size, h2)
      if (self$n_layers >= 3L) {
        h3 <- as.integer(max(floor(h2/2), floor_w))
        self$fc3 <- nn_linear(h2, h3)
        if (self$n_layers >= 4L) {
          h4 <- as.integer(max(floor(h3/2), floor_w))
          self$fc4 <- nn_linear(h3, h4); self$fc_out <- nn_linear(h4, L)
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
single_pred_net <- nn_module(
  "SinglePredNet",
  initialize = function(L_in, hidden_size = 8L, n_layers = 1L, p_dropout = 0.2) {
    self$n_layers <- as.integer(n_layers); self$act <- nn_relu()
    self$drop <- nn_dropout(p = p_dropout)
    self$fc1 <- nn_linear(L_in, hidden_size)
    if (self$n_layers >= 2L) {
      h2 <- as.integer(max(floor(hidden_size/2), 4L))
      self$fc2 <- nn_linear(hidden_size, h2)
      if (self$n_layers >= 3L) {
        h3 <- as.integer(max(floor(h2/2), 4L))
        self$fc3 <- nn_linear(h2, h3)
        if (self$n_layers >= 4L) {
          h4 <- as.integer(max(floor(h3/2), 4L))
          self$fc4 <- nn_linear(h3, h4); self$fc_out <- nn_linear(h4, 1L)
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
fanam_midas_module <- nn_module(
  "FANAM_MIDAS",
  initialize = function(d, K, L, r, H_agg, n_agg_layers,
                        H_pred, n_pred_layers, H_factor, n_factor_layers,
                        p_dropout, p_group_dropout = 0,
                        use_neural_agg = TRUE, use_residual = FALSE) {
    self$K <- as.integer(K); self$L <- as.integer(L); self$r <- as.integer(r)
    self$use_neural_agg <- use_neural_agg; self$use_residual <- use_residual
    self$p_group_dropout <- p_group_dropout
    self$agg_nets <- nn_module_list(
      lapply(seq_len(K), function(k)
        single_agg_net(as.integer(d), as.integer(L),
                       hidden_size = as.integer(H_agg),
                       n_layers = as.integer(n_agg_layers))))
    self$factor_proj <- NULL
    H_factor <- as.integer(H_factor)
    if (as.integer(n_factor_layers) >= 2L) {
      h2f <- as.integer(max(floor(H_factor/2), 4L))
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
      lapply(seq_len(K), function(k)
        single_pred_net(as.integer(L),
                        hidden_size = as.integer(H_pred),
                        n_layers = as.integer(n_pred_layers),
                        p_dropout = p_dropout)))
    self$bias <- nn_parameter(torch_zeros(1L))
  },
  set_factor_projection = function(P) {
    self$factor_proj <- torch_tensor(as.matrix(P), dtype = torch_float())
  },
  forward = function(Z_lags, W_fixed = NULL) {
    h_list <- vector("list", self$K); contrib_list <- vector("list", self$K)
    for (k in seq_len(self$K)) {
      h_k <- self$agg_nets[[k]](Z_lags[[k]])
      h_list[[k]] <- h_k
      contrib_list[[k]] <- self$pred_nets[[k]](h_k)$squeeze(dim = 2)
    }
    h_stacked <- torch_cat(h_list, dim = 2)
    f_tilde <- torch_mm(h_stacked,
                        self$factor_proj$to(device = h_stacked$device))
    factor_contrib <- self$factor_net(f_tilde)$squeeze(dim = 2)
    stacked <- torch_stack(contrib_list, dim = 2)
    factor_contrib + stacked$sum(dim = 2) + self$bias
  }
)

`%||%` <- function(a, b) if (is.null(a)) b else a

rebuild_state <- function(saved_list) {
  out <- list()
  if (!is.null(saved_list$bias))
    out[["bias"]] <- torch_tensor(saved_list$bias, dtype = torch_float())
  for (k in seq_along(saved_list$agg_nets %||% list())) {
    for (p in names(saved_list$agg_nets[[k]])) {
      out[[sprintf("agg_nets.%d.%s", k - 1L, p)]] <-
        torch_tensor(saved_list$agg_nets[[k]][[p]], dtype = torch_float())
    }
  }
  for (k in seq_along(saved_list$pred_nets %||% list())) {
    for (p in names(saved_list$pred_nets[[k]])) {
      out[[sprintf("pred_nets.%d.%s", k - 1L, p)]] <-
        torch_tensor(saved_list$pred_nets[[k]][[p]], dtype = torch_float())
    }
  }
  if (!is.null(saved_list$factor_net)) {
    for (p in names(saved_list$factor_net)) {
      out[[sprintf("factor_net.%s", p)]] <-
        torch_tensor(saved_list$factor_net[[p]], dtype = torch_float())
    }
  }
  out
}

n_h <- length(ths); n_it <- length(its)
predictions <- array(NA_real_, dim = c(n_cs, n_h, n_it, 2),
                     dimnames = list(cs_tickers,
                                     paste0("t=", ths),
                                     paste0("iter", its),
                                     c("rs1", "rs2")))

data_financial <- read.csv2(DATA_FILE)
end_observation <- '2020-12-31'
data_financial$start_day    <- sapply(data_financial$start_day, convert_to_ymd)
data_financial$censoringtime <- as.numeric(as.Date(end_observation) - as.Date(data_financial$start_day)) / 365
data_financial$Ts     <- data_financial$survival_time
data_financial$time   <- pmin(data_financial$Ts, data_financial$censoringtime)
data_financial$status <- data_financial$status

lookup <- read.csv2(file.path(RESULTS_DIR, "period_final_us_nonfin_lookup.csv"),
                    stringsAsFactors = FALSE)
CASE_STUDY_TICKERS <- c("AAPL", "MSFT", "ENRNQ", "KODK", "CCTYQ", "GM")
holdout_idx <- which(lookup$tic %in% CASE_STUDY_TICKERS)
data <- data_financial[-holdout_idx, , drop = FALSE]
cat(sprintf("\nUsing %d-firm training panel (case-study firms held out)\n", nrow(data)))

for (i_idx in seq_along(its)) {
  iter <- its[i_idx]
  for (h_idx in seq_along(ths)) {
    t_horizon <- ths[h_idx]

    nam_file <- file.path(RESULTS_DIR,
      sprintf("nam_holdout_worker_s%d_t%s_iter%02d.rds", s, t_horizon, iter))
    if (!file.exists(nam_file)) {
      cat(sprintf("  SKIP (no worker file): iter=%d t=%s\n", iter, t_horizon)); next
    }
    worker <- readRDS(nam_file)

    set.seed(s * iter); torch_manual_seed(s * iter)
    dataset_p <- data[which((1*(data$time<=t_horizon) * 1*(data$status==1)) == 1), ]
    dataset_n <- data[which((1*(data$time<=t_horizon) * 1*(data$status==1)) == 0), ]
    indices_p <- sample(nrow(dataset_p), nrow(dataset_p) * 0.8)
    indices_n <- sample(nrow(dataset_n), nrow(dataset_n) * 0.8)
    train_in <- rbind(dataset_p[indices_p, ], dataset_n[indices_n, ])
    train_dataset <- KM_estimate(testRandomLogitDataset(train_in, t = t_horizon))
    X_train <- apply(train_dataset[, 1:(num_vars * jmax)], 2, as.numeric)

    Z_train  <- extract_lag_groups(X_train, num_vars, jmax)
    tr_means <- lapply(Z_train, function(x) colMeans(x, na.rm = TRUE))
    tr_sds   <- lapply(Z_train, function(x) {
      ss <- apply(x, 2, sd, na.rm = TRUE); ss[ss < 1e-8] <- 1; ss
    })

    Z_cs     <- extract_lag_groups(cs_X, num_vars, jmax)
    Z_cs_std <- standardise_with_stats(Z_cs, tr_means, tr_sds)
    Z_cs_t   <- lapply(Z_cs_std, function(x)
                       torch_tensor(as.matrix(x), dtype = torch_float()))

    for (r in seq_along(worker$fanam_state_dicts)) {
      spec  <- worker$fanam_arch_specs[[r]]
      state <- worker$fanam_state_dicts[[r]]
      P     <- worker$fanam_pca_matrices[[r]]
      m <- fanam_midas_module(
        d = jmax, K = spec$K, L = spec$L, r = spec$r,
        H_agg = spec$H_agg, n_agg_layers = spec$n_agg_layers,
        H_pred = spec$H_pred, n_pred_layers = spec$n_pred_layers,
        H_factor = spec$H_factor, n_factor_layers = spec$n_factor_layers,
        p_dropout = spec$p_dropout,
        p_group_dropout = spec$p_group_dropout,
        use_neural_agg = spec$use_neural_agg,
        use_residual = spec$use_residual)
      m$load_state_dict(rebuild_state(state))
      m$set_factor_projection(P)
      m$eval()
      with_no_grad({
        probs <- as.numeric(torch_sigmoid(m(Z_cs_t))$cpu())
      })
      predictions[, h_idx, i_idx, r] <- probs
    }
    cat(sprintf("  done: iter=%2d t=%-3s (mean p across firms: %.3f)\n",
                iter, format(t_horizon, nsmall = 1),
                mean(predictions[, h_idx, i_idx, ], na.rm = TRUE)))
  }
}

cat("\n=================================================\n")
cat(" CASE STUDY SUMMARY (mean predicted distress prob)\n")
cat("=================================================\n\n")

summary_tab <- data.frame(
  ticker = cs_tickers,
  firm   = cs_firms,
  status = cs_status,
  event  = cs_bk,
  p_t8   = round(rowMeans(predictions[, 1, , ], na.rm = TRUE, dims = 1), 4),
  p_t85  = round(rowMeans(predictions[, 2, , ], na.rm = TRUE, dims = 1), 4),
  p_t9   = round(rowMeans(predictions[, 3, , ], na.rm = TRUE, dims = 1), 4),
  stringsAsFactors = FALSE
)
print(summary_tab, row.names = FALSE)

write.csv(summary_tab, "case_study_predictions.csv", row.names = FALSE)
saveRDS(predictions, "case_study_predictions_full.rds")
cat("\nSaved case_study_predictions.csv and case_study_predictions_full.rds\n")
