#' =============================================================================
#' Causal Effect Validation Experiment - Static (Single Time Point)
#' =============================================================================
#'
#' This script validates that the nonparanormal approximation correctly preserves
#' the causal margin p(Y|do(X)) by:
#' 1. Generating data with a KNOWN causal effect
#' 2. Estimating the causal effect using standard methods (IPW, G-computation, DR)
#' 3. Comparing estimates to the known truth
#'
#' DGP Structure:
#'   Z (confounder) --> X (treatment)
#'          |               |
#'          +--------> Y <--+
#'
#' Key insight: If the nonparanormal approximation correctly preserves p(Y|do(X)),
#' then IPW/DR estimators should be unbiased for the true ATE.
#' =============================================================================

library(tidyverse)
library(ggplot2)

# Source the nonparanormal functions
source("nonparanormal.R")
source("R/generate_frugal_outcome.R")

# =============================================================================
# Experiment Parameters
# =============================================================================

# Simulation settings
N_SAMPLES <- 5000        # Sample size per simulation
N_SIMS <- 200            # Number of Monte Carlo simulations
SEED_BASE <- 42          # Base seed for reproducibility

# Known causal parameters (GROUND TRUTH)
TRUE_BETA0 <- 0.0        # E[Y|do(X=0)]
TRUE_BETA1 <- 0.5        # Causal effect of X on Y
TRUE_ATE <- TRUE_BETA1   # ATE = E[Y|do(X=1)] - E[Y|do(X=0)] = beta1
CAUSAL_SD <- 1.0         # Std dev of Y|do(X)

# Confounding parameters
ALPHA0 <- -0.5           # Propensity intercept
ALPHA1 <- 0.8            # Confounding strength (higher = more confounding)

# Copula parameter: Y-Z partial correlation
RHO_Y_Z <- 0.6           # Dependence between Y and Z (conditional on X)

# Confounder distribution
Z_SHAPE <- 2             # Gamma shape
Z_SCALE <- 2             # Gamma scale

# Define causal mean function
causal_mean_fn <- function(x) TRUE_BETA0 + TRUE_BETA1 * x

# =============================================================================
# Helper Functions
# =============================================================================

#' Run a single simulation
#'
#' @param sim_id Simulation ID for seeding
#' @param n Sample size
#' @param verbose Print progress
#'
#' @return A data frame with one row containing all estimates
run_single_simulation <- function(sim_id, n = N_SAMPLES, verbose = FALSE) {
  set.seed(SEED_BASE + sim_id)

  # ---------------------------------------------------------------------------
  # Step 1: Generate Confounder Z
  # ---------------------------------------------------------------------------
  Z <- rgamma(n, shape = Z_SHAPE, scale = Z_SCALE)
  Z_mat <- as.matrix(Z)

  # ---------------------------------------------------------------------------
  # Step 2: Generate Treatment X | Z (creates confounding)
  # ---------------------------------------------------------------------------
  # Standardize Z for propensity model
  Z_std <- scale(Z)
  ps_true <- plogis(ALPHA0 + ALPHA1 * Z_std)
  X <- rbinom(n, 1, ps_true)

  # ---------------------------------------------------------------------------
  # Step 3: Generate Y using nonparanormal with known causal margin
  # ---------------------------------------------------------------------------
  frugal_result <- generate_frugal_outcome(
    X = X,
    Z = Z_mat,
    causal_mean_fn = causal_mean_fn,
    causal_sd = CAUSAL_SD,
    rho_Y_Z = RHO_Y_Z,
    seed = SEED_BASE + sim_id + 10000
  )
  Y <- frugal_result$Y

  # ---------------------------------------------------------------------------
  # Step 4: Estimate propensity scores (correctly specified)
  # ---------------------------------------------------------------------------
  ps_result <- estimate_propensity_scores(X, Z_mat)
  ps_hat <- ps_result$ps

  # ---------------------------------------------------------------------------
  # Step 5: Compute causal effect estimates
  # ---------------------------------------------------------------------------

  # Naive OLS (expected to be biased due to confounding)
  naive_result <- compute_naive_ate(Y, X)

  # IPW estimator
  ipw_result <- compute_ipw_ate(Y, X, ps_hat, stabilized = TRUE)

  # G-computation (outcome regression)
  gcomp_result <- compute_gcomp_ate(Y, X, Z_mat)

  # Doubly robust (AIPW)
  aipw_result <- compute_aipw_ate(Y, X, Z_mat, ps_hat)

  # ---------------------------------------------------------------------------
  # Step 6: Return results
  # ---------------------------------------------------------------------------
  results <- data.frame(
    sim_id = sim_id,
    n = n,
    true_ate = TRUE_ATE,
    naive_ate = naive_result$ate,
    ipw_ate = ipw_result$ate,
    gcomp_ate = gcomp_result$ate,
    aipw_ate = aipw_result$ate,
    # Also store potential outcome estimates
    ipw_mu1 = ipw_result$mu1,
    ipw_mu0 = ipw_result$mu0,
    gcomp_mu1 = gcomp_result$mu1,
    gcomp_mu0 = gcomp_result$mu0,
    aipw_mu1 = aipw_result$mu1,
    aipw_mu0 = aipw_result$mu0,
    # Diagnostics
    prop_treated = mean(X),
    mean_ps = mean(ps_hat),
    stringsAsFactors = FALSE
  )

  if (verbose) {
    cat(sprintf("Sim %d: True=%.3f, Naive=%.3f, IPW=%.3f, GComp=%.3f, AIPW=%.3f\n",
                sim_id, TRUE_ATE, naive_result$ate, ipw_result$ate,
                gcomp_result$ate, aipw_result$ate))
  }

  return(results)
}

# =============================================================================
# Run Simulation Study
# =============================================================================

cat("=============================================================================\n")
cat("Causal Effect Validation Experiment - Static Model\n")
cat("=============================================================================\n")
cat(sprintf("Sample size: %d\n", N_SAMPLES))
cat(sprintf("Number of simulations: %d\n", N_SIMS))
cat(sprintf("True ATE: %.3f\n", TRUE_ATE))
cat(sprintf("Confounding strength (alpha1): %.3f\n", ALPHA1))
cat(sprintf("Y-Z copula correlation: %.3f\n", RHO_Y_Z))
cat("=============================================================================\n\n")

# Run all simulations
cat("Running simulations...\n")
pb <- txtProgressBar(min = 0, max = N_SIMS, style = 3)

results_list <- vector("list", N_SIMS)
for (i in 1:N_SIMS) {
  results_list[[i]] <- run_single_simulation(i, verbose = FALSE)
  setTxtProgressBar(pb, i)
}
close(pb)

# Combine results
results_df <- do.call(rbind, results_list)

# =============================================================================
# Summarize Results
# =============================================================================

cat("\n=============================================================================\n")
cat("RESULTS SUMMARY\n")
cat("=============================================================================\n\n")

# Compute summary statistics
summary_stats <- results_df %>%
  summarise(
    # Naive
    naive_mean = mean(naive_ate),
    naive_bias = mean(naive_ate - true_ate),
    naive_sd = sd(naive_ate),
    naive_rmse = sqrt(mean((naive_ate - true_ate)^2)),
    # IPW
    ipw_mean = mean(ipw_ate),
    ipw_bias = mean(ipw_ate - true_ate),
    ipw_sd = sd(ipw_ate),
    ipw_rmse = sqrt(mean((ipw_ate - true_ate)^2)),
    # G-computation
    gcomp_mean = mean(gcomp_ate),
    gcomp_bias = mean(gcomp_ate - true_ate),
    gcomp_sd = sd(gcomp_ate),
    gcomp_rmse = sqrt(mean((gcomp_ate - true_ate)^2)),
    # AIPW
    aipw_mean = mean(aipw_ate),
    aipw_bias = mean(aipw_ate - true_ate),
    aipw_sd = sd(aipw_ate),
    aipw_rmse = sqrt(mean((aipw_ate - true_ate)^2))
  )

# Print formatted results
cat(sprintf("True ATE: %.4f\n\n", TRUE_ATE))

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
# Create Visualization
# =============================================================================

# Prepare data for plotting
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

# Create boxplot
p_boxplot <- ggplot(plot_data, aes(x = estimator, y = estimate, fill = estimator)) +
  geom_boxplot(alpha = 0.7) +
  geom_hline(yintercept = TRUE_ATE, linetype = "dashed", color = "red", linewidth = 1) +
  annotate("text", x = 0.5, y = TRUE_ATE + 0.05,
           label = sprintf("True ATE = %.2f", TRUE_ATE),
           hjust = 0, color = "red", size = 4) +
  labs(
    x = "Estimator",
    y = "Estimated ATE",
    title = "Causal Effect Estimates Across Simulations",
    subtitle = sprintf("N = %d samples, %d simulations | Confounding: alpha1 = %.1f | Copula rho = %.1f",
                       N_SAMPLES, N_SIMS, ALPHA1, RHO_Y_Z)
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold", size = 14),
    axis.title = element_text(face = "bold")
  ) +
  scale_fill_brewer(palette = "Set2")

# Create density plot
p_density <- ggplot(plot_data, aes(x = estimate, fill = estimator, color = estimator)) +
  geom_density(alpha = 0.3) +
  geom_vline(xintercept = TRUE_ATE, linetype = "dashed", color = "red", linewidth = 1) +
  facet_wrap(~ estimator, ncol = 2, scales = "free_y") +
  labs(
    x = "Estimated ATE",
    y = "Density",
    title = "Distribution of Causal Effect Estimates",
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

# =============================================================================
# Save Results
# =============================================================================

# Create output directory if it doesn't exist
if (!dir.exists("results")) {
  dir.create("results")
}

# Save plots
ggsave("results/causal_validation_static_boxplot.png", p_boxplot,
       width = 10, height = 6, dpi = 300)
ggsave("results/causal_validation_static_density.png", p_density,
       width = 10, height = 8, dpi = 300)

# Save results data
write.csv(results_df, "results/causal_validation_static_results.csv", row.names = FALSE)

# Save summary
summary_output <- list(
  parameters = list(
    n_samples = N_SAMPLES,
    n_sims = N_SIMS,
    true_ate = TRUE_ATE,
    alpha0 = ALPHA0,
    alpha1 = ALPHA1,
    rho_y_z = RHO_Y_Z,
    z_shape = Z_SHAPE,
    z_scale = Z_SCALE,
    causal_sd = CAUSAL_SD
  ),
  summary_stats = summary_stats,
  hypothesis_tests = list(
    ipw = list(t = ipw_ttest$statistic, p = ipw_ttest$p.value,
               ci = ipw_ttest$conf.int),
    gcomp = list(t = gcomp_ttest$statistic, p = gcomp_ttest$p.value,
                 ci = gcomp_ttest$conf.int),
    aipw = list(t = aipw_ttest$statistic, p = aipw_ttest$p.value,
                ci = aipw_ttest$conf.int)
  )
)
saveRDS(summary_output, "results/causal_validation_static_summary.rds")

cat("\n=============================================================================\n")
cat("Results saved to ./results/\n")
cat("  - causal_validation_static_boxplot.png\n")
cat("  - causal_validation_static_density.png\n")
cat("  - causal_validation_static_results.csv\n")
cat("  - causal_validation_static_summary.rds\n")
cat("=============================================================================\n")

# =============================================================================
# Verification Checklist
# =============================================================================

cat("\n")
cat("=============================================================================\n")
cat("VERIFICATION CHECKLIST\n")
cat("=============================================================================\n")

# Check 1: Naive OLS should be biased
naive_biased <- abs(summary_stats$naive_bias) > 0.05
cat(sprintf("[%s] Naive OLS is biased (|bias| = %.3f > 0.05)\n",
            ifelse(naive_biased, "PASS", "FAIL"), abs(summary_stats$naive_bias)))

# Check 2: IPW should be approximately unbiased
ipw_unbiased <- ipw_ttest$p.value > 0.05
cat(sprintf("[%s] IPW is unbiased (p = %.3f > 0.05)\n",
            ifelse(ipw_unbiased, "PASS", "WARN"), ipw_ttest$p.value))

# Check 3: G-computation should be approximately unbiased
gcomp_unbiased <- gcomp_ttest$p.value > 0.05
cat(sprintf("[%s] G-computation is unbiased (p = %.3f > 0.05)\n",
            ifelse(gcomp_unbiased, "PASS", "WARN"), gcomp_ttest$p.value))

# Check 4: AIPW should be approximately unbiased
aipw_unbiased <- aipw_ttest$p.value > 0.05
cat(sprintf("[%s] AIPW is unbiased (p = %.3f > 0.05)\n",
            ifelse(aipw_unbiased, "PASS", "WARN"), aipw_ttest$p.value))

# Check 5: Bias magnitude should be small for IPW/AIPW
ipw_small_bias <- abs(summary_stats$ipw_bias) < 0.05
aipw_small_bias <- abs(summary_stats$aipw_bias) < 0.05
cat(sprintf("[%s] IPW has small bias (|bias| = %.3f < 0.05)\n",
            ifelse(ipw_small_bias, "PASS", "WARN"), abs(summary_stats$ipw_bias)))
cat(sprintf("[%s] AIPW has small bias (|bias| = %.3f < 0.05)\n",
            ifelse(aipw_small_bias, "PASS", "WARN"), abs(summary_stats$aipw_bias)))

cat("=============================================================================\n")

# Print plots to screen
print(p_boxplot)
print(p_density)
