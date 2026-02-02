# =============================================================================
# Longitudinal Markov Model Experiment for Nonparanormal Approximation
# =============================================================================
#
# This experiment demonstrates that the nonparanormal approximation correctly
# imposes Markov properties in longitudinal causal models.
#
# Model Structure (per time point t = 1, ..., T_max):
#
#   Time t-1:  Z_1^{t-1} -----> Z_1^t
#                 |                |
#                 v                v
#              Z_2^{t-1} -----> Z_2^t
#                                  |
#                                  v
#                                 Y_t
#
# Markov Property:
#   Y_t _|_ (Z_1^{t-1}, Z_2^{t-1}, ...) | (Z_1^t, Z_2^t)
#
# =============================================================================

library(CondIndTests)
library(copula)
library(GeneralisedCovarianceMeasure)
library(ggplot2)
library(gridExtra)
library(ppcor)
library(tidyverse)
library(VineCopula)

source('nonparanormal.R')

# =============================================================================
# Helper Functions
# =============================================================================

#' Generate Longitudinal Confounders from Gamma Bayesian Network
#'
#' Generates confounders Z_1^t and Z_2^t for t = 1, ..., T_max using a non-linear
#' Gamma BN structure with temporal dependencies.
#'
#' @param T_max Number of time points
#' @param N Sample size
#' @param seed Random seed for reproducibility
#' @return A data frame with columns Z1_1, Z2_1, Z1_2, Z2_2, ..., Z1_T, Z2_T
generate_longitudinal_confounders <- function(T_max, N, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  data <- list()

  # Time t = 1: Initial distributions
  # Z_1^1 ~ Gamma(shape=2, scale=2)
  U1_1 <- runif(N)
  Z1_1 <- qgamma(U1_1, shape = 2, scale = 2)
  data[["Z1_1"]] <- Z1_1
  data[["U1_1"]] <- U1_1

  # Z_2^1 | Z_1^1 ~ Gamma(shape=2+1.5*Z_1^1, scale=1)
  U2_1 <- runif(N)
  Z2_1 <- qgamma(U2_1, shape = 2 + 1.5 * Z1_1, scale = 1)
  data[["Z2_1"]] <- Z2_1
  data[["U2_1"]] <- U2_1

  # Time t > 1: Temporal dependencies
  for (t in 2:T_max) {
    # Z_1^t | Z_1^{t-1} ~ Gamma(shape=2+1.5*Z_1^{t-1}, scale=1)
    U1_t <- runif(N)
    Z1_prev <- data[[paste0("Z1_", t-1)]]
    Z1_t <- qgamma(U1_t, shape = 2 + 1.5 * Z1_prev, scale = 1)
    data[[paste0("Z1_", t)]] <- Z1_t
    data[[paste0("U1_", t)]] <- U1_t

    # Z_2^t | Z_1^t, Z_2^{t-1} ~ Gamma(shape=2+Z_1^t+0.5*Z_2^{t-1}, scale=1)
    U2_t <- runif(N)
    Z2_prev <- data[[paste0("Z2_", t-1)]]
    Z2_t <- qgamma(U2_t, shape = 2 + Z1_t + 0.5 * Z2_prev, scale = 1)
    data[[paste0("Z2_", t)]] <- Z2_t
    data[[paste0("U2_", t)]] <- U2_t
  }

  return(as.data.frame(data))
}


#' Generate Outcome at Time t Using Nonparanormal Approximation
#'
#' Uses simulateConditionalOutcomeSamples() to generate Y_t based ONLY on
#' confounders at time t (Z_1^t, Z_2^t), thereby imposing the Markov property.
#'
#' @param Z1_t Confounder Z_1 at time t
#' @param Z2_t Confounder Z_2 at time t
#' @param U1_t Conditional rank for Z_1 at time t
#' @param U2_t Conditional rank for Z_2 at time t
#' @param rho_Y_Z2 Marginal correlation between Y_t and Z_2^t
#' @param rho_Y_Z1_given_Z2 Partial correlation between Y_t and Z_1^t given Z_2^t
#' @return A numeric vector of outcome values Y_t
generate_outcome_at_time <- function(Z1_t, Z2_t, U1_t, U2_t,
                                      rho_Y_Z2, rho_Y_Z1_given_Z2) {

  # Prepare covariate data and conditional ranks
  covariate_data <- data.frame(Z1 = Z1_t, Z2 = Z2_t)
  cond_covariate_ranks <- data.frame(U1 = U1_t, U2 = U2_t)

  # Topological order: Z2 first, then Z1
  # This means the vine structure links Y to Z2 first (marginal),
  # then Y to Z1 given Z2 (conditional)
  topoOrder <- c(2, 1)

  # Vine correlation parameters:
  # - First element: rho(Y, Z2) marginal
  # - Second element: rho(Y, Z1 | Z2) partial correlation
  vine_cor_params <- c(rho_Y_Z2, rho_Y_Z1_given_Z2)

  # Generate outcome using nonparanormal approximation
  sampleResults <- simulateConditionalOutcomeSamples(
    covariate_data = covariate_data,
    cond_covariate_ranks = cond_covariate_ranks,
    vine_cor_params = vine_cor_params,
    topoOrder = topoOrder
  )

  # Transform to standard Gaussian for interpretability
  outcomeRankSamples <- as.vector(sampleResults$outcomeRankSamples)
  Y_t <- qnorm(outcomeRankSamples)

  return(Y_t)
}


#' Compute Partial Correlation
#'
#' Computes the partial correlation between x and y given z
#'
#' @param x Numeric vector
#' @param y Numeric vector
#' @param z Numeric vector or matrix of conditioning variables
#' @return Partial correlation coefficient
compute_partial_cor <- function(x, y, z) {
  df <- data.frame(x = x, y = y, z = as.data.frame(z))
  result <- pcor.test(df$x, df$y, df[, -c(1,2), drop = FALSE])
  return(result$estimate)
}


# =============================================================================
# Plotting Theme
# =============================================================================

custom_theme <- theme_minimal() +
  theme(
    text = element_text(family = "Helvetica", size = 14),
    axis.title = element_text(face = "bold"),
    panel.grid.major = element_line(color = "grey90"),
    panel.grid.minor = element_line(color = "grey95"),
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white"),
    panel.border = element_blank(),
    plot.title = element_text(hjust = 0.5, face = "bold")
  )


# =============================================================================
# Main Experiment
# =============================================================================

run_markov_experiment <- function(
    T_max = 3,                    # Number of time points
    sample_sizes = c(1000, 5000, 20000),  # Sample sizes to test
    rho_Y_Z2 = 0.5,               # Marginal correlation Y - Z2
    rho_Y_Z1_given_Z2 = 0.8,      # Partial correlation Y - Z1 | Z2
    n_boot = 200,                 # Number of bootstrap iterations for tests
    boot_sample_size = 2000,      # Sample size per bootstrap iteration
    seed = 42,
    output_dir = "./plots/markov_results"
) {

  # Ensure output directory exists
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  # Storage for results
  all_results <- list()

  cat("\n========================================\n")
  cat("Longitudinal Markov Model Experiment\n")
  cat("========================================\n\n")
  cat("Parameters:\n")
  cat(sprintf("  T_max: %d time points\n", T_max))
  cat(sprintf("  rho(Y_t, Z_2^t): %.2f\n", rho_Y_Z2))
  cat(sprintf("  rho(Y_t, Z_1^t | Z_2^t): %.2f\n", rho_Y_Z1_given_Z2))
  cat(sprintf("  Bootstrap iterations: %d\n", n_boot))
  cat(sprintf("  Bootstrap sample size: %d\n\n", boot_sample_size))

  for (N in sample_sizes) {
    cat(sprintf("\n--- Sample Size N = %d ---\n\n", N))

    set.seed(seed)

    # -------------------------------------------------------------------------
    # Step 1: Generate longitudinal confounders
    # -------------------------------------------------------------------------
    cat("Generating longitudinal confounders...\n")
    data <- generate_longitudinal_confounders(T_max, N, seed = seed)

    # -------------------------------------------------------------------------
    # Step 2: Generate outcomes at each time point
    # -------------------------------------------------------------------------
    cat("Generating outcomes at each time point...\n")
    for (t in 1:T_max) {
      Z1_t <- data[[paste0("Z1_", t)]]
      Z2_t <- data[[paste0("Z2_", t)]]
      U1_t <- data[[paste0("U1_", t)]]
      U2_t <- data[[paste0("U2_", t)]]

      Y_t <- generate_outcome_at_time(
        Z1_t = Z1_t, Z2_t = Z2_t,
        U1_t = U1_t, U2_t = U2_t,
        rho_Y_Z2 = rho_Y_Z2,
        rho_Y_Z1_given_Z2 = rho_Y_Z1_given_Z2
      )

      data[[paste0("Y_", t)]] <- Y_t
    }

    # -------------------------------------------------------------------------
    # Step 3: Verify causal effects at each time point
    # -------------------------------------------------------------------------
    cat("\nVerifying causal effects:\n")
    effect_results <- list()

    for (t in 1:T_max) {
      Y_t <- data[[paste0("Y_", t)]]
      Z1_t <- data[[paste0("Z1_", t)]]
      Z2_t <- data[[paste0("Z2_", t)]]

      # Marginal correlation Y_t - Z2_t
      emp_rho_Y_Z2 <- cor(Y_t, Z2_t)

      # Partial correlation Y_t - Z1_t | Z2_t
      emp_rho_Y_Z1_given_Z2 <- compute_partial_cor(Y_t, Z1_t, Z2_t)

      effect_results[[paste0("t", t)]] <- list(
        rho_Y_Z2 = emp_rho_Y_Z2,
        rho_Y_Z1_given_Z2 = emp_rho_Y_Z1_given_Z2
      )

      cat(sprintf("  Time t=%d: rho(Y_t, Z_2^t) = %.3f (specified: %.2f)\n",
                  t, emp_rho_Y_Z2, rho_Y_Z2))
      cat(sprintf("           rho(Y_t, Z_1^t | Z_2^t) = %.3f (specified: %.2f)\n",
                  t, emp_rho_Y_Z1_given_Z2, rho_Y_Z1_given_Z2))
    }

    # -------------------------------------------------------------------------
    # Step 4: Conditional independence tests (Markov property)
    # -------------------------------------------------------------------------
    cat("\nTesting Markov property (conditional independence):\n")
    cond_ind_results <- list()

    for (t in 2:T_max) {
      cat(sprintf("\n  Time t=%d:\n", t))

      Y_t <- data[[paste0("Y_", t)]]
      Z1_t <- data[[paste0("Z1_", t)]]
      Z2_t <- data[[paste0("Z2_", t)]]
      Z1_prev <- data[[paste0("Z1_", t-1)]]
      Z2_prev <- data[[paste0("Z2_", t-1)]]

      # Scale variables for testing
      Y_t_scaled <- scale(Y_t)
      Z1_t_scaled <- scale(Z1_t)
      Z2_t_scaled <- scale(Z2_t)
      Z1_prev_scaled <- scale(Z1_prev)
      Z2_prev_scaled <- scale(Z2_prev)

      # Test 1: Y_t _|_ Z_1^{t-1} | (Z_1^t, Z_2^t)
      cat(sprintf("    Testing Y_%d _|_ Z_1^%d | (Z_1^%d, Z_2^%d)...\n", t, t-1, t, t))
      pvals_Y_Z1prev <- bootstrappedCondIndTest_GCM(
        x = as.vector(Y_t_scaled),
        y = as.vector(Z1_prev_scaled),
        z = cbind(Z1_t_scaled, Z2_t_scaled),
        n_boot = n_boot,
        sample_size = boot_sample_size,
        verbose = FALSE,
        regr.method = 'gam'
      )
      ks_Y_Z1prev <- ks.test(pvals_Y_Z1prev, "punif")

      # Test 2: Y_t _|_ Z_2^{t-1} | (Z_1^t, Z_2^t)
      cat(sprintf("    Testing Y_%d _|_ Z_2^%d | (Z_1^%d, Z_2^%d)...\n", t, t-1, t, t))
      pvals_Y_Z2prev <- bootstrappedCondIndTest_GCM(
        x = as.vector(Y_t_scaled),
        y = as.vector(Z2_prev_scaled),
        z = cbind(Z1_t_scaled, Z2_t_scaled),
        n_boot = n_boot,
        sample_size = boot_sample_size,
        verbose = FALSE,
        regr.method = 'gam'
      )
      ks_Y_Z2prev <- ks.test(pvals_Y_Z2prev, "punif")

      cond_ind_results[[paste0("t", t)]] <- list(
        pvals_Y_Z1prev = pvals_Y_Z1prev,
        ks_Y_Z1prev = ks_Y_Z1prev,
        pvals_Y_Z2prev = pvals_Y_Z2prev,
        ks_Y_Z2prev = ks_Y_Z2prev
      )

      cat(sprintf("      Y_%d _|_ Z_1^%d | Z^%d: KS p-value = %.4f\n",
                  t, t-1, t, ks_Y_Z1prev$p.value))
      cat(sprintf("      Y_%d _|_ Z_2^%d | Z^%d: KS p-value = %.4f\n",
                  t, t-1, t, ks_Y_Z2prev$p.value))
    }

    # -------------------------------------------------------------------------
    # Step 5: Marginal dependence tests (sanity check)
    # -------------------------------------------------------------------------
    cat("\nTesting marginal dependence (sanity check):\n")
    marg_dep_results <- list()

    for (t in 2:T_max) {
      cat(sprintf("\n  Time t=%d:\n", t))

      Y_t <- data[[paste0("Y_", t)]]
      Z1_prev <- data[[paste0("Z1_", t-1)]]
      Z2_prev <- data[[paste0("Z2_", t-1)]]

      # Scale variables
      Y_t_scaled <- scale(Y_t)
      Z1_prev_scaled <- scale(Z1_prev)
      Z2_prev_scaled <- scale(Z2_prev)

      # Test: Y_t vs Z_1^{t-1} (marginal)
      cat(sprintf("    Testing Y_%d vs Z_1^%d (marginal)...\n", t, t-1))
      pvals_marg_Z1 <- bootstrappedKendallTest(
        x = as.vector(Y_t_scaled),
        y = as.vector(Z1_prev_scaled),
        n_boot = n_boot,
        sample_size = boot_sample_size
      )
      ks_marg_Z1 <- ks.test(pvals_marg_Z1, "punif")

      # Test: Y_t vs Z_2^{t-1} (marginal)
      cat(sprintf("    Testing Y_%d vs Z_2^%d (marginal)...\n", t, t-1))
      pvals_marg_Z2 <- bootstrappedKendallTest(
        x = as.vector(Y_t_scaled),
        y = as.vector(Z2_prev_scaled),
        n_boot = n_boot,
        sample_size = boot_sample_size
      )
      ks_marg_Z2 <- ks.test(pvals_marg_Z2, "punif")

      marg_dep_results[[paste0("t", t)]] <- list(
        pvals_marg_Z1 = pvals_marg_Z1,
        ks_marg_Z1 = ks_marg_Z1,
        pvals_marg_Z2 = pvals_marg_Z2,
        ks_marg_Z2 = ks_marg_Z2
      )

      cat(sprintf("      Y_%d ~ Z_1^%d: KS p-value = %.4f (expect < 0.05)\n",
                  t, t-1, ks_marg_Z1$p.value))
      cat(sprintf("      Y_%d ~ Z_2^%d: KS p-value = %.4f (expect < 0.05)\n",
                  t, t-1, ks_marg_Z2$p.value))
    }

    # -------------------------------------------------------------------------
    # Store results for this sample size
    # -------------------------------------------------------------------------
    all_results[[paste0("N_", N)]] <- list(
      effect_results = effect_results,
      cond_ind_results = cond_ind_results,
      marg_dep_results = marg_dep_results
    )

    # -------------------------------------------------------------------------
    # Step 6: Generate plots
    # -------------------------------------------------------------------------
    cat("\nGenerating plots...\n")

    plot_list_cond_ind <- list()
    plot_list_marg_dep <- list()

    for (t in 2:T_max) {
      # Conditional independence plots (should be uniform)
      p1 <- ggplot(data.frame(p_value = cond_ind_results[[paste0("t", t)]]$pvals_Y_Z1prev),
                   aes(x = p_value)) +
        geom_histogram(bins = 10, fill = "steelblue", colour = "black", alpha = 0.7) +
        labs(title = bquote(Y[.(t)] ~ perp ~ Z[1]^{.(t-1)} ~ "|" ~ Z^{.(t)}),
             x = "p-value", y = "Frequency") +
        scale_x_continuous(limits = c(0, 1)) +
        custom_theme

      p2 <- ggplot(data.frame(p_value = cond_ind_results[[paste0("t", t)]]$pvals_Y_Z2prev),
                   aes(x = p_value)) +
        geom_histogram(bins = 10, fill = "steelblue", colour = "black", alpha = 0.7) +
        labs(title = bquote(Y[.(t)] ~ perp ~ Z[2]^{.(t-1)} ~ "|" ~ Z^{.(t)}),
             x = "p-value", y = "Frequency") +
        scale_x_continuous(limits = c(0, 1)) +
        custom_theme

      plot_list_cond_ind <- c(plot_list_cond_ind, list(p1, p2))

      # Marginal dependence plots (should NOT be uniform)
      p3 <- ggplot(data.frame(p_value = marg_dep_results[[paste0("t", t)]]$pvals_marg_Z1),
                   aes(x = p_value)) +
        geom_histogram(bins = 10, fill = "coral", colour = "black", alpha = 0.7) +
        labs(title = bquote(Y[.(t)] ~ "~" ~ Z[1]^{.(t-1)} ~ "(marginal)"),
             x = "p-value", y = "Frequency") +
        scale_x_continuous(limits = c(0, 1)) +
        custom_theme

      p4 <- ggplot(data.frame(p_value = marg_dep_results[[paste0("t", t)]]$pvals_marg_Z2),
                   aes(x = p_value)) +
        geom_histogram(bins = 10, fill = "coral", colour = "black", alpha = 0.7) +
        labs(title = bquote(Y[.(t)] ~ "~" ~ Z[2]^{.(t-1)} ~ "(marginal)"),
             x = "p-value", y = "Frequency") +
        scale_x_continuous(limits = c(0, 1)) +
        custom_theme

      plot_list_marg_dep <- c(plot_list_marg_dep, list(p3, p4))
    }

    # Combine and save conditional independence plots
    if (length(plot_list_cond_ind) > 0) {
      combined_cond_ind <- do.call(grid.arrange, c(plot_list_cond_ind, ncol = 2))
      ggsave(sprintf("cond_ind_N%d.png", N),
             plot = combined_cond_ind,
             path = output_dir,
             width = 10, height = 5 * (T_max - 1), dpi = 300)
    }

    # Combine and save marginal dependence plots
    if (length(plot_list_marg_dep) > 0) {
      combined_marg_dep <- do.call(grid.arrange, c(plot_list_marg_dep, ncol = 2))
      ggsave(sprintf("marg_dep_N%d.png", N),
             plot = combined_marg_dep,
             path = output_dir,
             width = 10, height = 5 * (T_max - 1), dpi = 300)
    }
  }

  # ===========================================================================
  # Summary Table
  # ===========================================================================
  cat("\n\n========================================\n")
  cat("SUMMARY RESULTS\n")
  cat("========================================\n\n")

  # Effect preservation summary
  cat("Causal Effect Verification:\n")
  cat("-----------------------------------------\n")
  cat(sprintf("%-10s | %-20s | %-20s\n", "N", "rho(Y_t, Z_2^t)", "rho(Y_t, Z_1^t|Z_2^t)"))
  cat(sprintf("%-10s | %-20s | %-20s\n", "", sprintf("(specified: %.2f)", rho_Y_Z2),
              sprintf("(specified: %.2f)", rho_Y_Z1_given_Z2)))
  cat("-----------------------------------------\n")

  for (N in sample_sizes) {
    effects <- all_results[[paste0("N_", N)]]$effect_results
    # Average across time points
    avg_rho_Y_Z2 <- mean(sapply(effects, function(x) x$rho_Y_Z2))
    avg_rho_Y_Z1_Z2 <- mean(sapply(effects, function(x) x$rho_Y_Z1_given_Z2))
    cat(sprintf("%-10d | %-20.3f | %-20.3f\n", N, avg_rho_Y_Z2, avg_rho_Y_Z1_Z2))
  }

  cat("\n\nMarkov Property Tests (Conditional Independence):\n")
  cat("p-values from KS test for uniformity (expect > 0.05):\n")
  cat("-----------------------------------------\n")
  cat(sprintf("%-10s | %-25s | %-25s\n", "N", "Y_t _|_ Z_1^{t-1} | Z^t", "Y_t _|_ Z_2^{t-1} | Z^t"))
  cat("-----------------------------------------\n")

  for (N in sample_sizes) {
    cond_ind <- all_results[[paste0("N_", N)]]$cond_ind_results
    # Collect KS p-values across time points
    ks_Z1_pvals <- sapply(cond_ind, function(x) x$ks_Y_Z1prev$p.value)
    ks_Z2_pvals <- sapply(cond_ind, function(x) x$ks_Y_Z2prev$p.value)
    cat(sprintf("%-10d | %-25s | %-25s\n", N,
                paste(sprintf("%.3f", ks_Z1_pvals), collapse=", "),
                paste(sprintf("%.3f", ks_Z2_pvals), collapse=", ")))
  }

  cat("\n\nMarginal Dependence Tests (Sanity Check):\n")
  cat("p-values from KS test for uniformity (expect < 0.05 = dependent):\n")
  cat("-----------------------------------------\n")
  cat(sprintf("%-10s | %-25s | %-25s\n", "N", "Y_t ~ Z_1^{t-1}", "Y_t ~ Z_2^{t-1}"))
  cat("-----------------------------------------\n")

  for (N in sample_sizes) {
    marg_dep <- all_results[[paste0("N_", N)]]$marg_dep_results
    # Collect KS p-values across time points
    ks_Z1_pvals <- sapply(marg_dep, function(x) x$ks_marg_Z1$p.value)
    ks_Z2_pvals <- sapply(marg_dep, function(x) x$ks_marg_Z2$p.value)
    cat(sprintf("%-10d | %-25s | %-25s\n", N,
                paste(sprintf("%.3f", ks_Z1_pvals), collapse=", "),
                paste(sprintf("%.3f", ks_Z2_pvals), collapse=", ")))
  }

  cat("\n========================================\n")
  cat("Experiment complete. Plots saved to:", output_dir, "\n")
  cat("========================================\n")

  return(all_results)
}


# =============================================================================
# Run the Experiment
# =============================================================================

# Set working directory to script location
if (interactive()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}

# Run with default parameters
results <- run_markov_experiment(
  T_max = 3,
  sample_sizes = c(1000, 5000, 20000),
  rho_Y_Z2 = 0.5,
  rho_Y_Z1_given_Z2 = 0.8,
  n_boot = 200,
  boot_sample_size = 2000,
  seed = 42
)

# Save results to file for later analysis
saveRDS(results, file = "./plots/markov_results/experiment_results.rds")
