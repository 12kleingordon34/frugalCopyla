# =============================================================================
# Longitudinal Markov Model with Treatment and Causal Margin
# =============================================================================
#
# This experiment extends the Markov model to include a treatment variable X_t
# and demonstrates that the nonparanormal approximation correctly:
#   1. Preserves the specified causal margin p(Y_t | do(X_t))
#   2. Imposes Markov properties in longitudinal models
#
# Model Structure (per time point t = 1, ..., T_max):
#
#   Time t-1:  Z_1^{t-1} -----> Z_1^t
#                 |                |
#                 v                v
#              Z_2^{t-1} -----> Z_2^t -----> X_t
#                                  |          |
#                                  v          v
#                                  +-----> Y_t
#
# Causal Margin:
#   Y_t | do(X_t) ~ Normal(X_t + 1, 1)
#
# Markov Property:
#   Y_t _|_ (Z_1^{t-1}, Z_2^{t-1}, ...) | (Z_1^t, Z_2^t, X_t)
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


#' Generate Treatment at Time t
#'
#' Generates treatment X_t as a function of confounders at time t.
#' X_t | Z_1^t, Z_2^t ~ Normal(0.5*Z_1^t + 0.3*Z_2^t, 1)
#'
#' @param Z1_t Confounder Z_1 at time t
#' @param Z2_t Confounder Z_2 at time t
#' @return A list with X_t values and their conditional ranks U_X
generate_treatment_at_time <- function(Z1_t, Z2_t) {
  N <- length(Z1_t)

  # Treatment depends on confounders
  # X_t | Z_1^t, Z_2^t ~ Normal(0.5*Z_1^t + 0.3*Z_2^t, 1)
  mean_X <- 0.5 * Z1_t + 0.3 * Z2_t
  U_X <- runif(N)
  X_t <- qnorm(U_X, mean = mean_X, sd = 1)

  return(list(X = X_t, U_X = U_X))
}


#' Generate Outcome at Time t with Causal Margin and Confounder Dependence
#'
#' Generates Y_t such that:
#'   1. The causal margin is preserved: Y_t | do(X_t) ~ Normal(X_t + 1, 1)
#'   2. Y_t depends on confounders Z^t via a Gaussian copula
#'   3. Markov property holds: Y_t _|_ Z^{t-1} | (Z^t, X_t)
#'
#' The approach:
#'   - Start with the causal margin: epsilon ~ Normal(0, 1), Y = X + 1 + epsilon
#'   - Link epsilon to confounders via Gaussian copula
#'   - This preserves E[Y | do(X)] = X + 1 and Var[Y | do(X)] = 1
#'
#' @param X_t Treatment at time t
#' @param Z1_t Confounder Z_1 at time t
#' @param Z2_t Confounder Z_2 at time t
#' @param U1_t Conditional rank for Z_1 at time t
#' @param U2_t Conditional rank for Z_2 at time t
#' @param rho_eps_Z2 Correlation between epsilon and Z_2^t (controls confounding)
#' @param rho_eps_Z1_given_Z2 Partial correlation between epsilon and Z_1^t | Z_2^t
#' @return A list with Y_t values and the error term epsilon
generate_outcome_with_causal_margin <- function(X_t, Z1_t, Z2_t, U1_t, U2_t,
                                                  rho_eps_Z2 = 0.4,
                                                  rho_eps_Z1_given_Z2 = 0.3) {

  # Prepare covariate data and conditional ranks for the copula
  # Note: We use confounders to generate the copula dependence
  covariate_data <- data.frame(Z1 = Z1_t, Z2 = Z2_t)
  cond_covariate_ranks <- data.frame(U1 = U1_t, U2 = U2_t)

  # Topological order: Z2 first, then Z1
  # This controls the vine factorization
  topoOrder <- c(2, 1)

  # Vine correlation parameters for epsilon-confounder dependence:
  # - rho(epsilon, Z2) marginal
  # - rho(epsilon, Z1 | Z2) partial correlation
  vine_cor_params <- c(rho_eps_Z2, rho_eps_Z1_given_Z2)

  # Generate epsilon (the error term) with copula dependence on confounders
  # This gives us a standard normal epsilon that is correlated with Z^t
  sampleResults <- simulateConditionalOutcomeSamples(
    covariate_data = covariate_data,
    cond_covariate_ranks = cond_covariate_ranks,
    vine_cor_params = vine_cor_params,
    topoOrder = topoOrder
  )

  # epsilon ~ Normal(0, 1) but correlated with confounders
  epsilon_ranks <- as.vector(sampleResults$outcomeRankSamples)
  epsilon <- qnorm(epsilon_ranks)

  # Apply the causal margin: Y_t | do(X_t) ~ Normal(X_t + 1, 1)
  # Y_t = X_t + 1 + epsilon
  # Under intervention do(X_t = x):
  #   - epsilon is independent of X_t (since copula only links to Z)
  #   - E[Y_t | do(X_t = x)] = x + 1 + E[epsilon] = x + 1
  #   - Var[Y_t | do(X_t = x)] = Var[epsilon] = 1
  Y_t <- X_t + 1 + epsilon

  return(list(
    Y = Y_t,
    epsilon = epsilon,
    epsilon_ranks = epsilon_ranks,
    fullCorrelationMatrix = sampleResults$fullCorrelationMatrix
  ))
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

run_markov_treatment_experiment <- function(
    T_max = 3,                    # Number of time points
    sample_sizes = c(1000, 5000, 20000),  # Sample sizes to test
    rho_eps_Z2 = 0.4,             # Correlation between epsilon and Z2 (confounding)
    rho_eps_Z1_given_Z2 = 0.3,    # Partial correlation epsilon-Z1 | Z2
    n_boot = 200,                 # Number of bootstrap iterations for tests
    boot_sample_size = 2000,      # Sample size per bootstrap iteration
    seed = 42,
    output_dir = "./plots/markov_treatment_results"
) {

  # Ensure output directory exists
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  # Storage for results
  all_results <- list()

  cat("\n========================================\n")
  cat("Longitudinal Markov Model with Treatment\n")
  cat("========================================\n\n")
  cat("Parameters:\n")
  cat(sprintf("  T_max: %d time points\n", T_max))
  cat("  Causal Margin: Y_t | do(X_t) ~ Normal(X_t + 1, 1)\n")
  cat(sprintf("  rho(epsilon, Z_2^t): %.2f (confounding strength)\n", rho_eps_Z2))
  cat(sprintf("  rho(epsilon, Z_1^t | Z_2^t): %.2f\n", rho_eps_Z1_given_Z2))
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
    # Step 2: Generate treatment and outcomes at each time point
    # -------------------------------------------------------------------------
    cat("Generating treatments and outcomes at each time point...\n")
    for (t in 1:T_max) {
      Z1_t <- data[[paste0("Z1_", t)]]
      Z2_t <- data[[paste0("Z2_", t)]]
      U1_t <- data[[paste0("U1_", t)]]
      U2_t <- data[[paste0("U2_", t)]]

      # Generate treatment
      treatment_result <- generate_treatment_at_time(Z1_t, Z2_t)
      X_t <- treatment_result$X
      data[[paste0("X_", t)]] <- X_t
      data[[paste0("UX_", t)]] <- treatment_result$U_X

      # Generate outcome with causal margin Y | do(X) ~ Normal(X + 1, 1)
      outcome_result <- generate_outcome_with_causal_margin(
        X_t = X_t, Z1_t = Z1_t, Z2_t = Z2_t,
        U1_t = U1_t, U2_t = U2_t,
        rho_eps_Z2 = rho_eps_Z2,
        rho_eps_Z1_given_Z2 = rho_eps_Z1_given_Z2
      )

      data[[paste0("Y_", t)]] <- outcome_result$Y
      data[[paste0("epsilon_", t)]] <- outcome_result$epsilon
    }

    # -------------------------------------------------------------------------
    # Step 3: Verify causal margin at each time point
    # -------------------------------------------------------------------------
    cat("\nVerifying causal margin preservation:\n")
    causal_margin_results <- list()

    for (t in 1:T_max) {
      Y_t <- data[[paste0("Y_", t)]]
      X_t <- data[[paste0("X_", t)]]
      epsilon_t <- data[[paste0("epsilon_", t)]]

      # The causal margin is Y | do(X) ~ Normal(X + 1, 1)
      # Residual should be: Y - X - 1 = epsilon ~ Normal(0, 1)
      residual <- Y_t - X_t - 1

      # Test that residual is standard normal
      mean_resid <- mean(residual)
      sd_resid <- sd(residual)

      # Shapiro-Wilk test on a subsample (max 5000)
      subsample <- residual[sample(length(residual), min(5000, length(residual)))]
      shapiro_test <- shapiro.test(subsample)

      causal_margin_results[[paste0("t", t)]] <- list(
        mean_residual = mean_resid,
        sd_residual = sd_resid,
        shapiro_pvalue = shapiro_test$p.value
      )

      cat(sprintf("  Time t=%d: E[Y - X - 1] = %.4f (expect: 0)\n", t, mean_resid))
      cat(sprintf("           SD[Y - X - 1] = %.4f (expect: 1)\n", sd_resid))
      cat(sprintf("           Shapiro-Wilk p = %.4f\n", shapiro_test$p.value))
    }

    # -------------------------------------------------------------------------
    # Step 4: Verify confounding (observational vs causal)
    # -------------------------------------------------------------------------
    cat("\nVerifying confounding structure:\n")
    confounding_results <- list()

    for (t in 1:T_max) {
      Y_t <- data[[paste0("Y_", t)]]
      X_t <- data[[paste0("X_", t)]]
      Z1_t <- data[[paste0("Z1_", t)]]
      Z2_t <- data[[paste0("Z2_", t)]]

      # Observational regression: Y ~ X (confounded)
      obs_model <- lm(Y_t ~ X_t)
      obs_coef <- coef(obs_model)["X_t"]

      # Causal effect should be 1 (from Y = X + 1 + epsilon)
      # Observational coefficient will be biased due to confounding

      confounding_results[[paste0("t", t)]] <- list(
        obs_coefficient = obs_coef,
        true_causal_effect = 1.0
      )

      cat(sprintf("  Time t=%d: Observational coef(Y~X) = %.3f (true causal: 1.0)\n",
                  t, obs_coef))
      cat(sprintf("           Bias = %.3f (due to confounding)\n", obs_coef - 1.0))
    }

    # -------------------------------------------------------------------------
    # Step 5: Conditional independence tests (Markov property)
    # -------------------------------------------------------------------------
    cat("\nTesting Markov property (conditional independence):\n")
    cat("  H0: Y_t _|_ Z^{t-1} | (Z^t, X_t)\n\n")
    cond_ind_results <- list()

    for (t in 2:T_max) {
      cat(sprintf("  Time t=%d:\n", t))

      Y_t <- data[[paste0("Y_", t)]]
      X_t <- data[[paste0("X_", t)]]
      Z1_t <- data[[paste0("Z1_", t)]]
      Z2_t <- data[[paste0("Z2_", t)]]
      Z1_prev <- data[[paste0("Z1_", t-1)]]
      Z2_prev <- data[[paste0("Z2_", t-1)]]

      # Scale variables for testing
      Y_t_scaled <- scale(Y_t)
      X_t_scaled <- scale(X_t)
      Z1_t_scaled <- scale(Z1_t)
      Z2_t_scaled <- scale(Z2_t)
      Z1_prev_scaled <- scale(Z1_prev)
      Z2_prev_scaled <- scale(Z2_prev)

      # Conditioning set includes current confounders AND treatment
      conditioning_set <- cbind(Z1_t_scaled, Z2_t_scaled, X_t_scaled)

      # Test 1: Y_t _|_ Z_1^{t-1} | (Z_1^t, Z_2^t, X_t)
      cat(sprintf("    Testing Y_%d _|_ Z_1^%d | (Z^%d, X_%d)...\n", t, t-1, t, t))
      pvals_Y_Z1prev <- bootstrappedCondIndTest_GCM(
        x = as.vector(Y_t_scaled),
        y = as.vector(Z1_prev_scaled),
        z = conditioning_set,
        n_boot = n_boot,
        sample_size = boot_sample_size,
        verbose = FALSE,
        regr.method = 'gam'
      )
      ks_Y_Z1prev <- ks.test(pvals_Y_Z1prev, "punif")

      # Test 2: Y_t _|_ Z_2^{t-1} | (Z_1^t, Z_2^t, X_t)
      cat(sprintf("    Testing Y_%d _|_ Z_2^%d | (Z^%d, X_%d)...\n", t, t-1, t, t))
      pvals_Y_Z2prev <- bootstrappedCondIndTest_GCM(
        x = as.vector(Y_t_scaled),
        y = as.vector(Z2_prev_scaled),
        z = conditioning_set,
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

      cat(sprintf("      Y_%d _|_ Z_1^%d | (Z^%d, X_%d): KS p-value = %.4f\n",
                  t, t-1, t, t, ks_Y_Z1prev$p.value))
      cat(sprintf("      Y_%d _|_ Z_2^%d | (Z^%d, X_%d): KS p-value = %.4f\n",
                  t, t-1, t, t, ks_Y_Z2prev$p.value))
    }

    # -------------------------------------------------------------------------
    # Step 6: Marginal dependence tests (sanity check)
    # -------------------------------------------------------------------------
    cat("\nTesting marginal dependence (sanity check):\n")
    cat("  Expect: Y_t depends on Z^{t-1} marginally (through the chain)\n\n")
    marg_dep_results <- list()

    for (t in 2:T_max) {
      cat(sprintf("  Time t=%d:\n", t))

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
      causal_margin_results = causal_margin_results,
      confounding_results = confounding_results,
      cond_ind_results = cond_ind_results,
      marg_dep_results = marg_dep_results
    )

    # -------------------------------------------------------------------------
    # Step 7: Generate plots
    # -------------------------------------------------------------------------
    cat("\nGenerating plots...\n")

    plot_list_cond_ind <- list()
    plot_list_marg_dep <- list()

    for (t in 2:T_max) {
      # Conditional independence plots (should be uniform)
      p1 <- ggplot(data.frame(p_value = cond_ind_results[[paste0("t", t)]]$pvals_Y_Z1prev),
                   aes(x = p_value)) +
        geom_histogram(bins = 10, fill = "steelblue", colour = "black", alpha = 0.7) +
        labs(title = bquote(Y[.(t)] ~ perp ~ Z[1]^{.(t-1)} ~ "|" ~ (Z^{.(t)} * "," ~ X[.(t)])),
             x = "p-value", y = "Frequency") +
        scale_x_continuous(limits = c(0, 1)) +
        custom_theme

      p2 <- ggplot(data.frame(p_value = cond_ind_results[[paste0("t", t)]]$pvals_Y_Z2prev),
                   aes(x = p_value)) +
        geom_histogram(bins = 10, fill = "steelblue", colour = "black", alpha = 0.7) +
        labs(title = bquote(Y[.(t)] ~ perp ~ Z[2]^{.(t-1)} ~ "|" ~ (Z^{.(t)} * "," ~ X[.(t)])),
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

    # Causal margin verification plot
    t_example <- 1
    Y_ex <- data[[paste0("Y_", t_example)]]
    X_ex <- data[[paste0("X_", t_example)]]
    residual_ex <- Y_ex - X_ex - 1

    p_causal <- ggplot(data.frame(residual = residual_ex), aes(x = residual)) +
      geom_histogram(aes(y = after_stat(density)), bins = 50,
                     fill = "darkgreen", colour = "black", alpha = 0.7) +
      stat_function(fun = dnorm, args = list(mean = 0, sd = 1),
                    color = "red", linewidth = 1.2, linetype = "dashed") +
      labs(title = "Causal Margin Verification: Y - X - 1 vs N(0,1)",
           x = "Residual (Y - X - 1)", y = "Density") +
      custom_theme

    ggsave(sprintf("causal_margin_N%d.png", N),
           plot = p_causal,
           path = output_dir,
           width = 8, height = 5, dpi = 300)
  }

  # ===========================================================================
  # Summary Table
  # ===========================================================================
  cat("\n\n========================================\n")
  cat("SUMMARY RESULTS\n")
  cat("========================================\n\n")

  # Causal margin verification
  cat("Causal Margin Verification: Y_t | do(X_t) ~ Normal(X_t + 1, 1)\n")
  cat("-----------------------------------------\n")
  cat(sprintf("%-10s | %-15s | %-15s | %-15s\n", "N", "E[Y-X-1]", "SD[Y-X-1]", "Shapiro p"))
  cat(sprintf("%-10s | %-15s | %-15s | %-15s\n", "", "(expect: 0)", "(expect: 1)", "(expect: >0.05)"))
  cat("-----------------------------------------\n")

  for (N in sample_sizes) {
    causal <- all_results[[paste0("N_", N)]]$causal_margin_results
    # Average across time points
    avg_mean <- mean(sapply(causal, function(x) x$mean_residual))
    avg_sd <- mean(sapply(causal, function(x) x$sd_residual))
    avg_shapiro <- mean(sapply(causal, function(x) x$shapiro_pvalue))
    cat(sprintf("%-10d | %-15.4f | %-15.4f | %-15.4f\n", N, avg_mean, avg_sd, avg_shapiro))
  }

  # Confounding verification
  cat("\n\nConfounding Verification (Observational vs Causal):\n")
  cat("-----------------------------------------\n")
  cat(sprintf("%-10s | %-20s | %-15s\n", "N", "Obs. coef(Y~X)", "Bias"))
  cat(sprintf("%-10s | %-20s | %-15s\n", "", "(true causal: 1.0)", ""))
  cat("-----------------------------------------\n")

  for (N in sample_sizes) {
    confound <- all_results[[paste0("N_", N)]]$confounding_results
    avg_obs_coef <- mean(sapply(confound, function(x) x$obs_coefficient))
    avg_bias <- avg_obs_coef - 1.0
    cat(sprintf("%-10d | %-20.4f | %-15.4f\n", N, avg_obs_coef, avg_bias))
  }

  # Markov property tests
  cat("\n\nMarkov Property Tests (Conditional Independence):\n")
  cat("p-values from KS test for uniformity (expect > 0.05):\n")
  cat("-----------------------------------------\n")
  cat(sprintf("%-10s | %-25s | %-25s\n", "N", "Y_t _|_ Z_1^{t-1} | ...", "Y_t _|_ Z_2^{t-1} | ..."))
  cat("-----------------------------------------\n")

  for (N in sample_sizes) {
    cond_ind <- all_results[[paste0("N_", N)]]$cond_ind_results
    ks_Z1_pvals <- sapply(cond_ind, function(x) x$ks_Y_Z1prev$p.value)
    ks_Z2_pvals <- sapply(cond_ind, function(x) x$ks_Y_Z2prev$p.value)
    cat(sprintf("%-10d | %-25s | %-25s\n", N,
                paste(sprintf("%.3f", ks_Z1_pvals), collapse=", "),
                paste(sprintf("%.3f", ks_Z2_pvals), collapse=", ")))
  }

  # Marginal dependence tests
  cat("\n\nMarginal Dependence Tests (Sanity Check):\n")
  cat("p-values from KS test for uniformity (expect < 0.05 = dependent):\n")
  cat("-----------------------------------------\n")
  cat(sprintf("%-10s | %-25s | %-25s\n", "N", "Y_t ~ Z_1^{t-1}", "Y_t ~ Z_2^{t-1}"))
  cat("-----------------------------------------\n")

  for (N in sample_sizes) {
    marg_dep <- all_results[[paste0("N_", N)]]$marg_dep_results
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
results <- run_markov_treatment_experiment(
  T_max = 3,
  sample_sizes = c(1000, 5000, 20000),
  rho_eps_Z2 = 0.4,
  rho_eps_Z1_given_Z2 = 0.3,
  n_boot = 200,
  boot_sample_size = 2000,
  seed = 42
)

# Save results to file for later analysis
saveRDS(results, file = "./plots/markov_treatment_results/experiment_results.rds")
