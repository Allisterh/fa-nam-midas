# Procrustes alignment of FA-NAM-MIDAS shape functions.
#
# Re-instantiates the trained FA-NAM-MIDAS restarts on the full panel,
# aligns the K per-covariate aggregation maps g_k against a reference
# restart by orthogonal Procrustes rotation, applies the rotation to the
# shape functions f_k, and averages across restarts to produce
# population-level shape figures and per-covariate signal-to-noise
# diagnostics.
#
# Set the input paths and `worker_pattern` before sourcing; the script
# works on any panel of FA-NAM-MIDAS worker files.
#
# Inputs
#   results_dir/<worker_pattern>          one worker file per iteration
#   data_dir/<data_file>                  the raw panel
#   data_dir/import functions for empirical application.R
# Outputs (out_dir)
#   population_shape_t<H>.pdf, top<N>_shape_t<H>.pdf,
#   shape_snr_t<H>.{csv,pdf}, alignment_diag_t<H>.csv,
#   aligned_data_t<H>.rds, and per-covariate 3D/pairwise figures.

if (!exists("results_dir"))     results_dir     <- "results"
if (!exists("data_dir"))        data_dir        <- "."
if (!exists("data_file"))       data_file       <- "panel.csv"
if (!exists("worker_pattern"))  worker_pattern  <- "nam_worker_s%d_t%s_iter%02d.rds"
if (!exists("out_dir"))         out_dir         <- "figures"
if (!exists("horizons"))        horizons        <- c(8, 8.5, 9)
if (!exists("s_value"))         s_value         <- 6
if (!exists("ref_pick"))        ref_pick        <- "max_cv_auc"
if (!exists("n_grid"))          n_grid          <- 200L
if (!exists("n_top"))           n_top           <- 6L
if (!exists("pop3d_top_n"))     pop3d_top_n     <- 3L
if (!exists("exclude_tickers")) exclude_tickers <- NULL
if (!exists("lookup_file"))     lookup_file     <- NULL

library(torch)
library(Survivalml)
library(survival)
library(ggplot2)
library(dplyr)
library(tidyr)
has_plot3D    <- requireNamespace("plot3D",    quietly = TRUE)
has_patchwork <- requireNamespace("patchwork", quietly = TRUE)
has_gridExtra <- requireNamespace("gridExtra", quietly = TRUE)

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
torch_set_num_threads(2L)

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

list_to_tensor_dict <- function(sd_list) {
  if (is.null(sd_list)) return(NULL)
  lapply(sd_list, function(arr) torch_tensor(arr, dtype = torch_float()))
}

reload_subnet <- function(constructor, state_dict_arrays) {
  net <- constructor()
  net$load_state_dict(list_to_tensor_dict(state_dict_arrays))
  net$eval()
  net
}

reload_fanam_restart <- function(state_dict, arch, pca_matrix) {
  K     <- arch$K
  L     <- arch$L
  d     <- arch$d
  H_agg <- arch$H_agg
  na_l  <- arch$n_agg_layers
  H_prd <- arch$H_pred
  np_l  <- arch$n_pred_layers
  pdr   <- arch$p_dropout

  agg_nets <- vector("list", K)
  for (k in seq_len(K)) {
    agg_nets[[k]] <- reload_subnet(
      function() single_agg_net(d_in = d, L = L,
                                hidden_size = H_agg, n_layers = na_l),
      state_dict$agg_nets[[k]]
    )
  }
  pred_nets <- vector("list", K)
  for (k in seq_len(K)) {
    pred_nets[[k]] <- reload_subnet(
      function() single_pred_net(L_in = L, hidden_size = H_prd,
                                 n_layers = np_l, p_dropout = pdr),
      state_dict$pred_nets[[k]]
    )
  }
  list(agg_nets = agg_nets, pred_nets = pred_nets,
       P = if (!is.null(pca_matrix)) as.matrix(pca_matrix) else NULL,
       arch = arch)
}

source(file.path(data_dir, "import functions for empirical application.R"))

load_full_panel <- function(t_horizon) {
  data_financial <- read.csv2(file.path(data_dir, data_file))
  p_fin <- ncol(data_financial) - 5
  for (col in names(data_financial)[seq_len(p_fin)]) {
    if (grepl("%", col)) data_financial[[col]] <- data_financial[[col]] * 0.01
  }
  data <- data_financial
  data$start_day     <- sapply(data$start_day, convert_to_ymd)
  data$censoringtime <- as.numeric(as.Date("2020-12-31") -
                                     as.Date(data$start_day)) / 365
  data$Ts     <- data$survival_time
  data$time   <- pmin(data$Ts, data$censoringtime)
  data$status <- data$status

  if (!is.null(exclude_tickers) && !is.null(lookup_file)) {
    lookup_path <- file.path(data_dir, lookup_file)
    if (file.exists(lookup_path)) {
      lookup <- read.csv2(lookup_path, stringsAsFactors = FALSE)
      drop_idx <- which(lookup$tic %in% exclude_tickers)
      if (length(drop_idx) > 0L) {
        data <- data[-drop_idx, , drop = FALSE]
        cat(sprintf("Excluded %d firms from the alignment panel.\n",
                    length(drop_idx)))
      }
    } else {
      warning(sprintf("Lookup file '%s' not found; no firms excluded.",
                      lookup_path))
    }
  }

  KM_estimate(testRandomLogitDataset(data, t = t_horizon))
}

extract_lag_groups_full <- function(panel_df, num_vars, jmax) {
  num_cols_to_use <- num_vars * jmax
  X <- apply(panel_df[, seq_len(num_cols_to_use)], 2, as.numeric)
  Z_lags <- vector("list", num_vars)
  for (k in seq_len(num_vars)) {
    Z_lags[[k]] <- X[, ((k - 1L) * jmax + 1L):(k * jmax), drop = FALSE]
  }
  Z_lags
}

standardise_full <- function(Z_lags) {
  lapply(Z_lags, function(M) {
    mu <- colMeans(M, na.rm = TRUE)
    sd <- apply(M, 2, sd, na.rm = TRUE); sd[sd < 1e-8] <- 1
    sweep(sweep(M, 2, mu, "-"), 2, sd, "/")
  })
}

eval_aggregations <- function(agg_nets, Z_lags_std) {
  with_no_grad({
    lapply(seq_along(agg_nets), function(k) {
      x <- torch_tensor(Z_lags_std[[k]], dtype = torch_float())
      h <- agg_nets[[k]](x)
      as.matrix(as.array(h$cpu()))
    })
  })
}

orthogonal_procrustes <- function(A, B) {
  sv <- svd(crossprod(A, B))
  sv$u %*% t(sv$v)
}

eval_pred_rotated <- function(pred_net, Q, h_grid_aligned) {
  h_in_iter_space <- h_grid_aligned %*% t(Q)
  with_no_grad({
    x <- torch_tensor(h_in_iter_space, dtype = torch_float())
    y <- pred_net(x)
    as.numeric(as.array(y$squeeze(dim = 2)$cpu()))
  })
}

eval_pred_direct <- function(pred_net, H_mat) {
  with_no_grad({
    x <- torch_tensor(H_mat, dtype = torch_float())
    y <- pred_net(x)$squeeze(dim = 2)
    as.numeric(as.array(y$cpu()))
  })
}

predictive_axis <- function(Hk_ref, pred_net_ref) {
  y_ref <- eval_pred_direct(pred_net_ref, Hk_ref)
  fit <- tryCatch(lm.fit(x = Hk_ref, y = y_ref), error = function(e) NULL)
  v <- if (is.null(fit) || any(!is.finite(fit$coefficients))) NULL else fit$coefficients
  if (is.null(v) || sqrt(sum(v^2)) < 1e-10) {
    pca <- prcomp(Hk_ref, center = FALSE)
    return(as.numeric(pca$rotation[, 1]))
  }
  as.numeric(v / sqrt(sum(v^2)))
}

run_horizon <- function(t_horizon) {
  cat(sprintf("\n=== Procrustes alignment for t = %s ===\n", t_horizon))

  iters <- list()
  for (i in 1:10) {
    fn <- file.path(results_dir,
                    sprintf(worker_pattern, s_value, t_horizon, i))
    if (file.exists(fn)) iters[[length(iters) + 1L]] <- readRDS(fn)
  }
  if (length(iters) == 0L) stop("No worker files found for t = ", t_horizon)
  cat(sprintf("Loaded %d iteration files.\n", length(iters)))

  panel_df <- load_full_panel(t_horizon)
  any_arch <- iters[[1]]$fanam_arch_specs[[1]]
  K        <- any_arch$K
  L        <- any_arch$L
  jmax     <- any_arch$d
  Z_lags   <- extract_lag_groups_full(panel_df, num_vars = K, jmax = jmax)
  Z_std    <- standardise_full(Z_lags)
  n_firms  <- nrow(Z_std[[1]])
  cat(sprintf("Panel size: n = %d firms; K = %d, L = %d.\n", n_firms, K, L))

  cv_aucs <- sapply(iters, function(x) {
    if (is.null(x$cv_auc_factor) || is.na(x$cv_auc_factor)) -Inf else x$cv_auc_factor
  })
  ref_idx <- if (is.numeric(ref_pick) && length(ref_pick) == 1L) {
    as.integer(ref_pick)
  } else if (ref_pick == "max_cv_auc") {
    which.max(cv_aucs)
  } else {
    which(cv_aucs == sort(cv_aucs)[ceiling(length(cv_aucs) / 2)])[1]
  }
  cat(sprintf("Reference iteration: %d (CV-AUC = %.3f)\n",
              ref_idx, cv_aucs[ref_idx]))

  restarts <- list()
  for (i in seq_along(iters)) {
    nrest <- length(iters[[i]]$fanam_state_dicts)
    for (rs in seq_len(nrest)) {
      restarts[[length(restarts) + 1L]] <- list(
        iter = i, restart = rs,
        is_ref = (i == ref_idx && rs == 1L),
        net = reload_fanam_restart(
          state_dict  = iters[[i]]$fanam_state_dicts[[rs]],
          arch        = iters[[i]]$fanam_arch_specs[[rs]],
          pca_matrix  = iters[[i]]$fanam_pca_matrices[[rs]]
        )
      )
    }
  }
  cat(sprintf("Reloaded %d restart models.\n", length(restarts)))

  H_list <- lapply(restarts, function(r) eval_aggregations(r$net$agg_nets, Z_std))

  ref_pos <- which(sapply(restarts, function(r) r$is_ref))[1]
  if (is.na(ref_pos)) ref_pos <- 1L
  H_ref <- H_list[[ref_pos]]

  diag_rows <- list()
  Q_list <- vector("list", length(restarts))
  for (j in seq_along(restarts)) {
    Q_list[[j]] <- vector("list", K)
    for (k in seq_len(K)) {
      Q <- orthogonal_procrustes(H_list[[j]][[k]], H_ref[[k]])
      Q_list[[j]][[k]] <- Q
      resid <- H_list[[j]][[k]] %*% Q - H_ref[[k]]
      diag_rows[[length(diag_rows) + 1L]] <- data.frame(
        iter = restarts[[j]]$iter, restart = restarts[[j]]$restart,
        covariate = k, procrustes_rms = sqrt(mean(resid^2)),
        stringsAsFactors = FALSE)
    }
  }
  diag_df <- do.call(rbind, diag_rows)
  cat(sprintf("Median Procrustes residual (RMS): %.4f\n",
              median(diag_df$procrustes_rms)))

  ref_pred_nets <- restarts[[ref_pos]]$net$pred_nets

  shape_rows <- list()
  for (k in seq_len(K)) {
    Hk_ref <- H_ref[[k]]
    axis_v <- predictive_axis(Hk_ref, ref_pred_nets[[k]])
    proj_ref <- as.numeric(Hk_ref %*% axis_v)
    grid_x <- seq(quantile(proj_ref, 0.01), quantile(proj_ref, 0.99),
                  length.out = n_grid)
    h_grid_ref <- outer(grid_x, axis_v)

    f_per_restart <- matrix(NA_real_, nrow = n_grid, ncol = length(restarts))
    for (j in seq_along(restarts)) {
      f_per_restart[, j] <- eval_pred_rotated(
        restarts[[j]]$net$pred_nets[[k]], Q_list[[j]][[k]], h_grid_ref)
    }
    f_mean <- rowMeans(f_per_restart, na.rm = TRUE)
    f_sd   <- apply(f_per_restart, 1, sd, na.rm = TRUE)
    shape_rows[[k]] <- data.frame(
      covariate = k, x = grid_x, f_mean = f_mean,
      f_lo = f_mean - f_sd, f_hi = f_mean + f_sd,
      stringsAsFactors = FALSE)
  }
  shape_df <- do.call(rbind, shape_rows)

  snr_df <- shape_df %>%
    group_by(covariate) %>%
    summarise(
      shape_range = max(f_mean) - min(f_mean),
      mean_ribbon = mean(f_hi - f_lo),
      shape_snr   = shape_range / pmax(mean_ribbon, 1e-6),
      mean_abs_f  = mean(abs(f_mean)),
      .groups = "drop"
    ) %>%
    arrange(desc(shape_snr))
  cat(sprintf("Top-%d covariates by shape SNR: %s\n", n_top,
              paste(snr_df$covariate[seq_len(min(n_top, nrow(snr_df)))],
                    collapse = ", ")))

  write.csv(diag_df,
            file.path(out_dir, sprintf("alignment_diag_t%s.csv", t_horizon)),
            row.names = FALSE)
  write.csv(snr_df,
            file.path(out_dir, sprintf("shape_snr_t%s.csv", t_horizon)),
            row.names = FALSE)
  saveRDS(list(shape_df = shape_df, diag_df = diag_df, snr_df = snr_df,
               ref_idx = ref_idx, K = K, L = L, n_firms = n_firms),
          file.path(out_dir, sprintf("aligned_data_t%s.rds", t_horizon)))

  p1 <- ggplot(shape_df, aes(x = x, y = f_mean)) +
    geom_ribbon(aes(ymin = f_lo, ymax = f_hi), fill = "#59a14f", alpha = 0.25) +
    geom_line(color = "#59a14f", linewidth = 0.6) +
    facet_wrap(~ covariate, scales = "free", ncol = 8) +
    labs(x = "h_k along reference PC1", y = expression(f[k](h[k]))) +
    theme_minimal(base_size = 9) +
    theme(panel.grid.minor = element_blank(),
          strip.text = element_text(size = 8))
  ggsave(file.path(out_dir, sprintf("population_shape_t%s.pdf", t_horizon)),
         p1, width = 16, height = ceiling(K / 8) * 1.7, limitsize = FALSE)

  p2 <- ggplot(diag_df, aes(x = factor(covariate), y = procrustes_rms)) +
    geom_boxplot(fill = "#76b7b2", alpha = 0.7, outlier.size = 0.6) +
    labs(x = "Covariate k", y = "RMS residual after rotation") +
    theme_minimal(base_size = 10) +
    theme(panel.grid.minor = element_blank())
  ggsave(file.path(out_dir, sprintf("residual_distribution_t%s.pdf", t_horizon)),
         p2, width = 12, height = 4)

  top_k <- snr_df$covariate[seq_len(min(n_top, nrow(snr_df)))]
  shape_top <- shape_df %>%
    filter(covariate %in% top_k) %>%
    mutate(covariate_lbl = sprintf("k = %d  (SNR = %.2f)",
                                   covariate,
                                   snr_df$shape_snr[match(covariate, snr_df$covariate)]),
           covariate_lbl = factor(covariate_lbl,
                                  levels = sprintf("k = %d  (SNR = %.2f)",
                                                   top_k,
                                                   snr_df$shape_snr[seq_along(top_k)])))
  p_top <- ggplot(shape_top, aes(x = x, y = f_mean)) +
    geom_ribbon(aes(ymin = f_lo, ymax = f_hi), fill = "#59a14f", alpha = 0.25) +
    geom_line(color = "#59a14f", linewidth = 0.9) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey60",
               linewidth = 0.4) +
    facet_wrap(~ covariate_lbl, scales = "free", ncol = 3) +
    labs(x = "h_k along predictive axis", y = expression(f[k](h[k]))) +
    theme_minimal(base_size = 11) +
    theme(panel.grid.minor = element_blank(),
          strip.text = element_text(face = "bold"))
  ggsave(file.path(out_dir, sprintf("top%d_shape_t%s.pdf",
                                    length(top_k), t_horizon)),
         p_top, width = 12, height = ceiling(length(top_k) / 3) * 3.0)

  snr_bar <- snr_df %>%
    mutate(in_top   = covariate %in% top_k,
           covariate = factor(covariate, levels = sort(unique(covariate))))
  p_snr <- ggplot(snr_bar, aes(x = covariate, y = shape_snr, fill = in_top)) +
    geom_col(alpha = 0.85) +
    scale_fill_manual(values = c(`TRUE` = "#59a14f", `FALSE` = "#bab0ac"),
                      guide = "none") +
    labs(x = "Covariate k", y = "Shape SNR") +
    theme_minimal(base_size = 10) +
    theme(panel.grid.minor = element_blank())
  ggsave(file.path(out_dir, sprintf("shape_snr_t%s.pdf", t_horizon)),
         p_snr, width = 12, height = 4)

  if (has_patchwork) {
    p_snr_shared <- p_snr +
      labs(x = NULL) +
      theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
    ggsave(file.path(out_dir, sprintf("snr_residuals_t%s.pdf", t_horizon)),
           patchwork::wrap_plots(p_snr_shared, p2, ncol = 1, heights = c(1, 1)),
           width = 12, height = 6.5)
  }

  pop_ks <- snr_df$covariate[seq_len(min(pop3d_top_n, nrow(snr_df)))]
  for (k in pop_ks) {
    Hk_ref <- H_ref[[k]]
    f_per_restart <- matrix(NA_real_, nrow = nrow(Hk_ref), ncol = length(restarts))
    for (j in seq_along(restarts)) {
      f_per_restart[, j] <- eval_pred_rotated(
        restarts[[j]]$net$pred_nets[[k]], Q_list[[j]][[k]], Hk_ref)
    }
    f_pop <- rowMeans(f_per_restart, na.rm = TRUE)

    if (has_plot3D) {
      pdf(file.path(out_dir, sprintf("pop3d_t%s_k%02d.pdf", t_horizon, k)),
          width = 6.5, height = 5.5)
      plot3D::scatter3D(
        x = Hk_ref[, 1], y = Hk_ref[, 2], z = Hk_ref[, 3], colvar = f_pop,
        pch = 19, cex = 0.7,
        col = plot3D::ramp.col(c("#3b4cc0", "#dddddd", "#b40426"), n = 100),
        bty = "b2", phi = 20, theta = 35, ticktype = "detailed",
        xlab = "h_{k,1}", ylab = "h_{k,2}", zlab = "h_{k,3}", clab = "f_k^pop")
      dev.off()
    }

    df_pair <- data.frame(h1 = Hk_ref[, 1], h2 = Hk_ref[, 2],
                          h3 = Hk_ref[, 3], f = f_pop)
    f_lim <- max(abs(f_pop), na.rm = TRUE)
    pair_specs <- list(
      list(xv = "h1", yv = "h2"),
      list(xv = "h1", yv = "h3"),
      list(xv = "h2", yv = "h3")
    )
    pair_plots <- lapply(pair_specs, function(spec) {
      ggplot(df_pair, aes(x = .data[[spec$xv]], y = .data[[spec$yv]], color = f)) +
        geom_point(size = 1.0, alpha = 0.85) +
        scale_color_gradient2(low = "#3b4cc0", mid = "#dddddd", high = "#b40426",
                              midpoint = 0, limits = c(-f_lim, f_lim),
                              name = expression(f[k]^{pop})) +
        labs(x = spec$xv, y = spec$yv) +
        theme_minimal(base_size = 10) +
        theme(panel.grid.minor = element_blank(), legend.position = "right")
    })
    out_pair <- file.path(out_dir, sprintf("poppair_t%s_k%02d.pdf", t_horizon, k))
    if (has_patchwork) {
      ggsave(out_pair,
             patchwork::wrap_plots(pair_plots, ncol = 3) +
               patchwork::plot_layout(guides = "collect"),
             width = 14, height = 4.5)
    } else if (has_gridExtra) {
      ggsave(out_pair, gridExtra::arrangeGrob(grobs = pair_plots, ncol = 3),
             width = 14, height = 4.5)
    } else {
      pdf(out_pair, width = 14, height = 4.5)
      print(pair_plots[[1]] + pair_plots[[2]] + pair_plots[[3]])
      dev.off()
    }
  }

  invisible(list(shape_df = shape_df, diag_df = diag_df, snr_df = snr_df))
}

for (t_h in horizons) run_horizon(t_h)
cat("\n=== Procrustes alignment finished. Output in", out_dir, "===\n")
