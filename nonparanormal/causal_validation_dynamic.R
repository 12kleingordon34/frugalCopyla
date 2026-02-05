#' =============================================================================
#' Causal Effect Validation Experiment - Dynamic (Two Time Points)
#' =============================================================================
#'
#' This script validates the nonparanormal approximation for longitudinal settings
#' with treatment at T=2. Tests whether p(Y2|do(X2), Y1) is correctly preserved.
#'
#' DGP Structure:
#'
#' Time 1:                     Time 2:
#' Z1 ──┬──> X1               Z2 ──┬──> X2
#'      └──> Y1 <── X1             └──> Y2 <── X2
#'                                       ↑
#'          Y1 ────────────────────────┘
#'
#' Key features:
#' - Treatment only at T=2 (X2 is the intervention of interest)
#' - Y1 acts as a confounder for the Y2 ~ X2 relationship
#' - Z2 depends on (Z1, Y1) - temporal dynamics
#'
#' =============================================================================

library(tidyverse)
library(ggplot2)
library(ppcor)

# Source the nonparanormal functions
source("nonparanormal.R")
source("R/generate_frugal_outcome.R")

# =============================================================================
# Experiment Parameters
# =============================================================================

# Simulation settings
N_SAMPLES <- 5000        # Sample size per simulation
N_SIMS <- 200            # Number of Monte Carlo simulations
SEED_BASE <- 123         # Base seed for reproducibility

# -----------------------------------------------------------------------------
# Time 1 Parameters
# -----------------------------------------------------------------------------
# Z1 distribution
Z1_SHAPE <- 2
Z1_SCALE <- 2

# X1 propensity (not used for causal estimation, just generates data)
ALPHA_X1_0 <- -0.3       # Intercept
ALPHA_X1_Z1 <- 0.5       # Z1 effect on X1

# Y1 causal parameters (Y1 | do(X1))
BETA_Y1_0 <- 0.0         # Intercept
BETA_Y1_X1 <- 0.4        # X1 effect on Y1
Y1_SD <- 1.0

# Y1-Z1 copula correlation
RHO_Y1_Z1 <- 0.5

# -----------------------------------------------------------------------------
# Time 2 Parameters
# -----------------------------------------------------------------------------
# Z2 | Z1, Y1 - temporal dynamics via Gamma BN
Z2_BASE_SHAPE <- 2
Z2_SCALE <- 1

# X2 propensity (the treatment of interest)
ALPHA_X2_0 <- -0.5       # Intercept
ALPHA_X2_Z2 <- 0.6       # Z2 effect on X2
ALPHA_X2_Y1 <- 0.4       # Y1 effect on X2 (confounding)

# Y2 causal parameters: E[Y2 | do(X2=x), Y1=y] = gamma0 + gamma1*x + gamma2*y
# This is the dynamic causal margin we want to preserve
GAMMA_Y2_0 <- 0.0        # Intercept
GAMMA_Y2_X2 <- 0.5       # CAUSAL EFFECT of X2 on Y2 (this is the target)
GAMMA_Y2_Y1 <- 0.3       # Effect of Y1 on Y2 (not intervened)
Y2_SD <- 1.0

# Y2-(Z2, Y1) copula correlations (partial correlations in topological order)
# Order: Z2, Y1, then Y2
RHO_Y2_Z2 <- 0.4         # Y2-Z2 partial correlation
RHO_Y2_Y1 <- 0.3         # Y2-Y1 partial correlation (given Z2)

# True ATE for X2 (marginalizing over Y1)
TRUE_ATE_X2 <- GAMMA_Y2_X2

# =============================================================================
# Helper Functions for Dynamic Setting
# =============================================================================

#' Generate data from the dynamic DGP (single time point approach)
#'
#' @param n Sample size
#' @param seed Random seed
#'
#' @return List with all generated data
generate_dynamic_data <- function(n, seed) {
  set.seed(seed)

  # -------------------------------------------------------------------------
  # TIME 1
  # -------------------------------------------------------------------------

  # Generate Z1
  U_Z1 <- runif(n)
  Z1 <- qgamma(U_Z1, shape = Z1_SHAPE, scale = Z1_SCALE)

  # Generate X1 | Z1
  Z1_std <- scale(Z1)
  ps_X1 <- plogis(ALPHA_X1_0 + ALPHA_X1_Z1 * Z1_std)
  X1 <- rbinom(n, 1, ps_X1)

  # Generate Y1 using frugal parameterization
  causal_mean_Y1 <- function(x) BETA_Y1_0 + BETA_Y1_X1 * x

  Y1_result <- generate_frugal_outcome(
    X = X1,
    Z = as.matrix(Z1),
    causal_mean_fn = causal_mean_Y1,
    causal_sd = Y1_SD,
    rho_Y_Z = RHO_Y1_Z1,
    seed = seed + 10000
  )
  Y1 <- Y1_result$Y

  # -------------------------------------------------------------------------
  # TIME 2
  # -------------------------------------------------------------------------

  # Generate Z2 | Z1, Y1 via Gamma BN
  # Shape depends on (Z1, Y1)
  U_Z2_given_Z1_Y1 <- runif(n)
  Z2_shape <- Z2_BASE_SHAPE + 0.5 * pmax(Z1, 0) + 0.3 * pmax(Y1, 0)
  Z2 <- qgamma(U_Z2_given_Z1_Y1, shape = Z2_shape, scale = Z2_SCALE)

  # Generate X2 | Z2, Y1 (the treatment of interest)
  Z2_std <- scale(Z2)
  Y1_std <- scale(Y1)
  ps_X2_true <- plogis(ALPHA_X2_0 + ALPHA_X2_Z2 * Z2_std + ALPHA_X2_Y1 * Y1_std)
  X2 <- rbinom(n, 1, ps_X2_true)

  # -------------------------------------------------------------------------
  # Generate Y2 using frugal parameterization
  # -------------------------------------------------------------------------
  # The causal margin is: E[Y2 | do(X2), Y1] = gamma0 + gamma1*X2 + gamma2*Y1
  # Note: Y1 is NOT intervened, so it enters the causal margin

  causal_mean_Y2 <- function(x2, y1) {
    GAMMA_Y2_0 + GAMMA_Y2_X2 * x2 + GAMMA_Y2_Y1 * y1
  }

  # Confounders for Y2 are (Z2, Y1)
  # We need to set up the copula structure carefully
  Z_for_Y2 <- cbind(Z2, Y1)

  # Convert to empirical ranks
  Z_ranks_Y2 <- apply(Z_for_Y2, 2, function(col) {
    rank(col, ties.method = "average") / (n + 1)
  })

  # Fit Gaussian copula to (Z2, Y1)
  fitted_copula_Y2 <- fitMVGaussianCopula(dataQuantiles = Z_ranks_Y2, method = 'itau')

  # Build full correlation matrix including Y2
  topoOrder_Y2 <- 1:2  # Z2, Y1 (then Y2 will be added)
  fullCorMatrix_Y2 <- computeFullCorMatrix(
    topoOrder = topoOrder_Y2,
    corMatrixMN = fitted_copula_Y2$correlationMatrix,
    vineCorParams = c(RHO_Y2_Z2, RHO_Y2_Y1)
  )

  # Generate Y2 conditional on (Z2, Y1)
  Z_normal_Y2 <- qnorm(Z_ranks_Y2)
  outcome_model_Y2 <- multivariate_conditional_mean_and_samples(
    X2_samples = Z_normal_Y2,
    R = fullCorMatrix_Y2
  )

  # Transform to have correct dynamic causal margin
  Y2_ranks <- pnorm(as.vector(outcome_model_Y2$generated_samples))
  causal_means_Y2 <- causal_mean_Y2(X2, Y1)
  Y2 <- causal_means_Y2 + Y2_SD * qnorm(Y2_ranks)

  # -------------------------------------------------------------------------
  # Return all data
  # -------------------------------------------------------------------------
  return(list(
    # Time 1
    Z1 = Z1,
    X1 = X1,
    Y1 = Y1,
    ps_X1 = ps_X1,
    # Time 2
    Z2 = Z2,
    X2 = X2,
    Y2 = Y2,
    ps_X2_true = ps_X2_true,
    # Copula info
    fitted_copula_Y2 = fitted_copula_Y2,
    fullCorMatrix_Y2 = fullCorMatrix_Y2
  ))
}


#' Estimate propensity scores for X2 given (Z2, Y1)
#'
#' @param X2 Treatment at time 2
#' @param Z2 Confounder at time 2
#' @param Y1 Outcome at time 1 (also a confounder for Y2)
#'
#' @return Estimated propensity scores
estimate_ps_X2 <- function(X2, Z2, Y1) {
  data <- data.frame(X2 = X2, Z2 = scale(Z2), Y1 = scale(Y1))
  model <- glm(X2 ~ Z2 + Y1, data = data, family = binomial())
  return(fitted(model))
}


#' Compute IPW estimate for dynamic ATE of X2
#'
#' @param Y2 Outcome at time 2
#' @param X2 Treatment at time 2
#' @param ps Propensity scores P(X2=1|Z2, Y1)
#'
#' @return IPW estimate of E[Y2|do(X2=1)] - E[Y2|do(X2=0)]
compute_ipw_dynamic <- function(Y2, X2, ps) {
  ps <- pmax(pmin(ps, 0.99), 0.01)

  # Stabilized weights
  p_X2 <- mean(X2)
  w1 <- p_X2 / ps
  w0 <- (1 - p_X2) / (1 - ps)

  mu1 <- sum(X2 * w1 * Y2) / sum(X2 * w1)
  mu0 <- sum((1 - X2) * w0 * Y2) / sum((1 - X2) * w0)

  return(list(ate = mu1 - mu0, mu1 = mu1, mu0 = mu0))
}


#' Compute G-computation estimate for dynamic ATE
#'
#' @param Y2 Outcome at time 2
#' @param X2 Treatment at time 2
#' @param Z2 Confounder at time 2
#' @param Y1 Outcome/confounder from time 1
#'
#' @return G-computation estimate
compute_gcomp_dynamic <- function(Y2, X2, Z2, Y1) {
  data <- data.frame(Y2 = Y2, X2 = X2, Z2 = scale(Z2), Y1 = scale(Y1))
  model <- lm(Y2 ~ X2 + Z2 + Y1, data = data)

  data_X2_1 <- data
  data_X2_1$X2 <- 1
  data_X2_0 <- data
  data_X2_0$X2 <- 0

  Y2_hat_1 <- predict(model, newdata = data_X2_1)
  Y2_hat_0 <- predict(model, newdata = data_X2_0)

  mu1 <- mean(Y2_hat_1)
  mu0 <- mean(Y2_hat_0)

  return(list(ate = mu1 - mu0, mu1 = mu1, mu0 = mu0, model = model))
}


#' Compute AIPW estimate for dynamic setting
#'
#' @param Y2 Outcome at time 2
#' @param X2 Treatment at time 2
#' @param Z2 Confounder at time 2
#' @param Y1 Outcome from time 1
#' @param ps Propensity scores
#'
#' @return AIPW estimate
compute_aipw_dynamic <- function(Y2, X2, Z2, Y1, ps) {
  ps <- pmax(pmin(ps, 0.99), 0.01)

  data <- data.frame(Y2 = Y2, X2 = X2, Z2 = scale(Z2), Y1 = scale(Y1))
  outcome_model <- lm(Y2 ~ X2 + Z2 + Y1, data = data)

  data_X2_1 <- data
  data_X2_1$X2 <- 1
  data_X2_0 <- data
  data_X2_0$X2 <- 0

  mu_hat_1 <- predict(outcome_model, newdata = data_X2_1)
  mu_hat_0 <- predict(outcome_model, newdata = data_X2_0)

  # AIPW
  phi_1 <- mu_hat_1 + X2 * (Y2 - mu_hat_1) / ps
  phi_0 <- mu_hat_0 + (1 - X2) * (Y2 - mu_hat_0) / (1 - ps)

  mu1 <- mean(phi_1)
  mu0 <- mean(phi_0)

  return(list(ate = mu1 - mu0, mu1 = mu1, mu0 = mu0))
}


#' Run a single dynamic simulation
#'
#' @param sim_id Simulation ID
#' @param n Sample size
#' @param verbose Print progress
#'
#' @return Data frame with results
run_single_dynamic_simulation <- function(sim_id, n = N_SAMPLES, verbose = FALSE) {
  # Generate data
  data <- generate_dynamic_data(n, seed = SEED_BASE + sim_id)

  # Estimate propensity scores for X2
  ps_X2_hat <- estimate_ps_X2(data$X2, data$Z2, data$Y1)

  # Naive estimate (ignoring confounding)
  naive_model <- lm(data$Y2 ~ data$X2)
  naive_ate <- coef(naive_model)["data$X2"]

  # IPW estimate
  ipw_result <- compute_ipw_dynamic(data$Y2, data$X2, ps_X2_hat)

  # G-computation estimate
  gcomp_result <- compute_gcomp_dynamic(data$Y2, data$X2, data$Z2, data$Y1)

  # AIPW estimate
  aipw_result <- compute_aipw_dynamic(data$Y2, data$X2, data$Z2, data$Y1, ps_X2_hat)

  # Markov property test
  markov_result <- test_markov_property(data)

  results <- data.frame(
    sim_id = sim_id,
    n = n,
    true_ate = TRUE_ATE_X2,
    naive_ate = as.numeric(naive_ate),
    ipw_ate = ipw_result$ate,
    gcomp_ate = gcomp_result$ate,
    aipw_ate = aipw_result$ate,
    # Markov test (partial correlation)
    pcor_Z1 = markov_result$pcor_Z1_estimate,
    pcor_Z1_z = markov_result$pcor_Z1_statistic,
    pcor_Z1_p = markov_result$pcor_Z1_pvalue,
    # Additional info
    prop_treated_X2 = mean(data$X2),
    mean_ps_X2 = mean(ps_X2_hat),
    cor_Z2_Y1 = cor(data$Z2, data$Y1),
    stringsAsFactors = FALSE
  )

  if (verbose) {
    cat(sprintf("Sim %d: True=%.3f, Naive=%.3f, IPW=%.3f, GComp=%.3f, AIPW=%.3f | Markov p=%.3f\n",
                sim_id, TRUE_ATE_X2, naive_ate, ipw_result$ate,
                gcomp_result$ate, aipw_result$ate, markov_result$pcor_Z1_pvalue))
  }

  return(results)
}


# =============================================================================
# Test Markov Property: Y2 ⊥ Z1 | Z2, X2, Y1
# =============================================================================

#' Test the Markov property for the dynamic model
#'
#' Under the correct specification, Y2 should be conditionally independent
#' of Z1 given (Z2, X2, Y1). This tests whether the nonparanormal approximation
#' preserves the correct conditional independence structure.
#'
#' Uses ppcor::pcor.test for consistency with the longitudinal validation.
#'
#' @param data Data list from generate_dynamic_data
#'
#' @return List with test results
test_markov_property <- function(data) {
  # Test: Y2 ⊥ Z1 | Z2, X2, Y1
  # Using partial correlation (consistent with longitudinal script)
  pcor_Z1 <- ppcor::pcor.test(
    data$Y2, data$Z1,
    cbind(data$Z2, data$X2, data$Y1)
  )

  return(list(
    pcor_Z1_estimate = pcor_Z1$estimate,
    pcor_Z1_statistic = pcor_Z1$statistic,
    pcor_Z1_pvalue = pcor_Z1$p.value,
    markov_holds = pcor_Z1$p.value > 0.05  # Fail to reject => Markov property holds
  ))
}


# =============================================================================
# Run Simulation Study
# =============================================================================

cat("=============================================================================\n")
cat("Causal Effect Validation Experiment - Dynamic Model (Two Time Points)\n")
cat("=============================================================================\n")
cat(sprintf("Sample size: %d\n", N_SAMPLES))
cat(sprintf("Number of simulations: %d\n", N_SIMS))
cat(sprintf("True ATE (X2 on Y2): %.3f\n", TRUE_ATE_X2))
cat(sprintf("Dynamic causal margin: E[Y2|do(X2), Y1] = %.1f + %.1f*X2 + %.1f*Y1\n",
            GAMMA_Y2_0, GAMMA_Y2_X2, GAMMA_Y2_Y1))
cat("=============================================================================\n\n")

# First, test Markov property on a single large dataset
cat("Testing Markov property (Y2 ⊥ Z1 | Z2, X2, Y1) on single large dataset...\n")
test_data <- generate_dynamic_data(10000, seed = 888)
markov_test <- test_markov_property(test_data)
cat(sprintf("  Partial correlation: %.4f (z = %.2f, p = %.4f)\n",
            markov_test$pcor_Z1_estimate, markov_test$pcor_Z1_statistic,
            markov_test$pcor_Z1_pvalue))
cat(sprintf("  Markov property: %s\n\n",
            ifelse(markov_test$markov_holds, "HOLDS (p > 0.05)", "VIOLATED")))

# Run all simulations
cat("Running simulations...\n")
pb <- txtProgressBar(min = 0, max = N_SIMS, style = 3)

results_list <- vector("list", N_SIMS)
for (i in 1:N_SIMS) {
  results_list[[i]] <- run_single_dynamic_simulation(i, verbose = FALSE)
  setTxtProgressBar(pb, i)
}
close(pb)

# Combine results
results_df <- do.call(rbind, results_list)

# =============================================================================
# Summarize Results
# =============================================================================

cat("\n=============================================================================\n")
cat("RESULTS SUMMARY - DYNAMIC MODEL\n")
cat("=============================================================================\n\n")

summary_stats <- results_df %>%
  summarise(
    naive_mean = mean(naive_ate),
    naive_bias = mean(naive_ate - true_ate),
    naive_sd = sd(naive_ate),
    naive_rmse = sqrt(mean((naive_ate - true_ate)^2)),
    ipw_mean = mean(ipw_ate),
    ipw_bias = mean(ipw_ate - true_ate),
    ipw_sd = sd(ipw_ate),
    ipw_rmse = sqrt(mean((ipw_ate - true_ate)^2)),
    gcomp_mean = mean(gcomp_ate),
    gcomp_bias = mean(gcomp_ate - true_ate),
    gcomp_sd = sd(gcomp_ate),
    gcomp_rmse = sqrt(mean((gcomp_ate - true_ate)^2)),
    aipw_mean = mean(aipw_ate),
    aipw_bias = mean(aipw_ate - true_ate),
    aipw_sd = sd(aipw_ate),
    aipw_rmse = sqrt(mean((aipw_ate - true_ate)^2))
  )

cat(sprintf("True ATE (X2 effect): %.4f\n\n", TRUE_ATE_X2))

cat("Estimator         | Mean    | Bias    | SD      | RMSE    \n")
cat("------------------|---------|---------|---------|----------\n")
cat(sprintf("Naive OLS         | %.4f  | %.4f  | %.4f  | %.4f  \n",
            summary_stats$naive_mean, summary_stats$naive_bias,
            summary_stats$naive_sd, summary_stats$naive_rmse))
cat(sprintf("IPW               | %.4f  | %.4f  | %.4f  | %.4f  \n",
            summary_stats$ipw_mean, summary_stats$ipw_bias,
            summary_stats$ipw_sd, summary_stats$ipw_rmse))
cat(sprintf("G-computation     | %.4f  | %.4f  | %.4f  | %.4f  \n",
            summary_stats$gcomp_mean, summary_stats$gcomp_bias,
            summary_stats$gcomp_sd, summary_stats$gcomp_rmse))
cat(sprintf("AIPW (DR)         | %.4f  | %.4f  | %.4f  | %.4f  \n",
            summary_stats$aipw_mean, summary_stats$aipw_bias,
            summary_stats$aipw_sd, summary_stats$aipw_rmse))

cat("\n")

# Statistical tests for bias
cat("Statistical tests (H0: bias = 0):\n")
cat("----------------------------------\n")

ipw_ttest <- t.test(results_df$ipw_ate - results_df$true_ate)
gcomp_ttest <- t.test(results_df$gcomp_ate - results_df$true_ate)
aipw_ttest <- t.test(results_df$aipw_ate - results_df$true_ate)

cat(sprintf("IPW:   t = %.3f, p = %.4f, 95%% CI = [%.4f, %.4f]\n",
            ipw_ttest$statistic, ipw_ttest$p.value,
            ipw_ttest$conf.int[1], ipw_ttest$conf.int[2]))
cat(sprintf("GComp: t = %.3f, p = %.4f, 95%% CI = [%.4f, %.4f]\n",
            gcomp_ttest$statistic, gcomp_ttest$p.value,
            gcomp_ttest$conf.int[1], gcomp_ttest$conf.int[2]))
cat(sprintf("AIPW:  t = %.3f, p = %.4f, 95%% CI = [%.4f, %.4f]\n",
            aipw_ttest$statistic, aipw_ttest$p.value,
            aipw_ttest$conf.int[1], aipw_ttest$conf.int[2]))

# =============================================================================
# Markov Property Summary
# =============================================================================

cat("\n")
cat("Markov Property Tests (across simulations):\n")
cat("--------------------------------------------\n")

markov_summary <- results_df %>%
  summarise(
    mean_pcor_Z1 = mean(pcor_Z1),
    sd_pcor_Z1 = sd(pcor_Z1),
    mean_z_Z1 = mean(pcor_Z1_z),
    mean_p_Z1 = mean(pcor_Z1_p),
    sd_p_Z1 = sd(pcor_Z1_p)
  )

cat(sprintf("Y2 ⊥ Z1 | Z2, X2, Y1:\n"))
cat(sprintf("  Mean partial corr: %.4f (SD: %.4f)\n",
            markov_summary$mean_pcor_Z1, markov_summary$sd_pcor_Z1))
cat(sprintf("  Mean z-statistic: %.2f\n", markov_summary$mean_z_Z1))
cat(sprintf("  Mean p-value: %.3f (SD: %.3f)\n",
            markov_summary$mean_p_Z1, markov_summary$sd_p_Z1))

# KS test for uniformity of p-values
ks_Z1 <- ks.test(results_df$pcor_Z1_p, "punif")

cat(sprintf("\nKS test for uniformity of p-values:\n"))
cat(sprintf("  Z1: p = %.3f\n", ks_Z1$p.value))

# =============================================================================
# Create Visualization
# =============================================================================

plot_data <- results_df %>%
  dplyr::select(sim_id, naive_ate, ipw_ate, gcomp_ate, aipw_ate) %>%
  pivot_longer(cols = ends_with("_ate"),
               names_to = "estimator",
               values_to = "estimate") %>%
  mutate(
    estimator = factor(estimator,
                       levels = c("naive_ate", "ipw_ate", "gcomp_ate", "aipw_ate"),
                       labels = c("Naive OLS", "IPW", "G-computation", "AIPW (DR)"))
  )

# Boxplot
p_boxplot <- ggplot(plot_data, aes(x = estimator, y = estimate, fill = estimator)) +
  geom_boxplot(alpha = 0.7) +
  geom_hline(yintercept = TRUE_ATE_X2, linetype = "dashed", color = "red", linewidth = 1) +
  annotate("text", x = 0.5, y = TRUE_ATE_X2 + 0.05,
           label = sprintf("True ATE = %.2f", TRUE_ATE_X2),
           hjust = 0, color = "red", size = 4) +
  labs(
    x = "Estimator",
    y = "Estimated ATE (X2 on Y2)",
    title = "Dynamic Model: Causal Effect Estimates",
    subtitle = sprintf("N = %d, %d sims | E[Y2|do(X2),Y1] = %.1f + %.1fX2 + %.1fY1",
                       N_SAMPLES, N_SIMS, GAMMA_Y2_0, GAMMA_Y2_X2, GAMMA_Y2_Y1)
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold", size = 14),
    axis.title = element_text(face = "bold")
  ) +
  scale_fill_brewer(palette = "Set2")

# Density plot
p_density <- ggplot(plot_data, aes(x = estimate, fill = estimator, color = estimator)) +
  geom_density(alpha = 0.3) +
  geom_vline(xintercept = TRUE_ATE_X2, linetype = "dashed", color = "red", linewidth = 1) +
  facet_wrap(~ estimator, ncol = 2, scales = "free_y") +
  labs(
    x = "Estimated ATE",
    y = "Density",
    title = "Distribution of Dynamic Causal Effect Estimates",
    subtitle = "Red dashed line = True ATE"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold", size = 14),
    axis.title = element_text(face = "bold")
  ) +
  scale_fill_brewer(palette = "Set2") +
  scale_color_brewer(palette = "Set2")

# Markov p-value histogram
p_markov_Z1 <- ggplot(results_df, aes(x = pcor_Z1_p)) +
  geom_histogram(bins = 20, fill = "steelblue", color = "white", alpha = 0.7) +
  geom_hline(yintercept = N_SIMS / 20, linetype = "dashed", color = "red") +
  labs(
    x = "p-value",
    y = "Frequency",
    title = expression(paste("Markov test: ", Y[2], " ⊥ ", Z[1], " | ", Z[2], ", ", X[2], ", ", Y[1])),
    subtitle = sprintf("KS test for uniformity: p = %.3f", ks_Z1$p.value)
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 12))

# =============================================================================
# Save Results
# =============================================================================

if (!dir.exists("results")) {
  dir.create("results")
}

ggsave("results/causal_validation_dynamic_boxplot.png", p_boxplot,
       width = 10, height = 6, dpi = 300)
ggsave("results/causal_validation_dynamic_density.png", p_density,
       width = 10, height = 8, dpi = 300)
ggsave("results/causal_validation_dynamic_markov_Z1.png", p_markov_Z1,
       width = 8, height = 5, dpi = 300)

write.csv(results_df, "results/causal_validation_dynamic_results.csv", row.names = FALSE)

summary_output <- list(
  parameters = list(
    n_samples = N_SAMPLES,
    n_sims = N_SIMS,
    true_ate_x2 = TRUE_ATE_X2,
    gamma_y2_0 = GAMMA_Y2_0,
    gamma_y2_x2 = GAMMA_Y2_X2,
    gamma_y2_y1 = GAMMA_Y2_Y1,
    rho_y2_z2 = RHO_Y2_Z2,
    rho_y2_y1 = RHO_Y2_Y1
  ),
  markov_test = markov_test,
  markov_summary = markov_summary,
  ks_tests = list(Z1_p = ks_Z1$p.value),
  summary_stats = summary_stats,
  hypothesis_tests = list(
    ipw = list(t = ipw_ttest$statistic, p = ipw_ttest$p.value),
    gcomp = list(t = gcomp_ttest$statistic, p = gcomp_ttest$p.value),
    aipw = list(t = aipw_ttest$statistic, p = aipw_ttest$p.value)
  )
)
saveRDS(summary_output, "results/causal_validation_dynamic_summary.rds")

cat("\n=============================================================================\n")
cat("Results saved to ./results/\n")
cat("  - causal_validation_dynamic_boxplot.png\n")
cat("  - causal_validation_dynamic_density.png\n")
cat("  - causal_validation_dynamic_markov_Z1.png\n")
cat("  - causal_validation_dynamic_results.csv\n")
cat("  - causal_validation_dynamic_summary.rds\n")
cat("=============================================================================\n")

# =============================================================================
# Verification Checklist
# =============================================================================

cat("\n")
cat("=============================================================================\n")
cat("VERIFICATION CHECKLIST - DYNAMIC MODEL\n")
cat("=============================================================================\n")

# Check 1: Markov property holds (p-values uniform via KS test)
markov_Z1_ok <- ks_Z1$p.value > 0.05
cat(sprintf("[%s] Markov Y2 ⊥ Z1 | cond (KS p = %.3f)\n",
            ifelse(markov_Z1_ok, "PASS", "WARN"), ks_Z1$p.value))

# Check 2: Naive OLS biased
naive_biased <- abs(summary_stats$naive_bias) > 0.05
cat(sprintf("[%s] Naive OLS is biased (|bias| = %.3f > 0.05)\n",
            ifelse(naive_biased, "PASS", "FAIL"), abs(summary_stats$naive_bias)))

# Check 3-5: Estimators approximately unbiased
ipw_unbiased <- ipw_ttest$p.value > 0.05
gcomp_unbiased <- gcomp_ttest$p.value > 0.05
aipw_unbiased <- aipw_ttest$p.value > 0.05

cat(sprintf("[%s] IPW is unbiased (p = %.3f > 0.05)\n",
            ifelse(ipw_unbiased, "PASS", "WARN"), ipw_ttest$p.value))
cat(sprintf("[%s] G-computation is unbiased (p = %.3f > 0.05)\n",
            ifelse(gcomp_unbiased, "PASS", "WARN"), gcomp_ttest$p.value))
cat(sprintf("[%s] AIPW is unbiased (p = %.3f > 0.05)\n",
            ifelse(aipw_unbiased, "PASS", "WARN"), aipw_ttest$p.value))

# Check 6-7: Small bias magnitude
ipw_small_bias <- abs(summary_stats$ipw_bias) < 0.05
aipw_small_bias <- abs(summary_stats$aipw_bias) < 0.05

cat(sprintf("[%s] IPW has small bias (|bias| = %.3f < 0.05)\n",
            ifelse(ipw_small_bias, "PASS", "WARN"), abs(summary_stats$ipw_bias)))
cat(sprintf("[%s] AIPW has small bias (|bias| = %.3f < 0.05)\n",
            ifelse(aipw_small_bias, "PASS", "WARN"), abs(summary_stats$aipw_bias)))

cat("=============================================================================\n")

# Print plots
print(p_boxplot)
print(p_density)
print(p_markov_Z1)
