#!/usr/bin/env Rscript
# =============================================================================
# Compare BN vs Nonparanormal Covariate Conditional Behaviour
# =============================================================================
#
# This script compares covariates from the original Gamma BN chain vs a full
# nonparanormal surrogate (Gaussian BN + marginal transforms). The goal is to
# see "first hand" how the conditional behaviour differs between the two.
#
# Dataset A: Original Gamma BN chain
#   Z1 ~ Gamma(shape=2, scale=2)
#   Z2 | Z1 ~ Gamma(shape = 1.5*Z1 + 2, scale=2)
#   Z3 | Z2 ~ Gamma(shape = 1.5*Z2 + 2, scale=2)
#
# Dataset B: Full nonparanormal surrogate
#   Latent Gaussian BN fitted to Dataset A, then margins mapped back via
#   empirical inverse CDFs to match Dataset A marginals exactly.
#
# Usage:
#   Rscript --no-init-file experiments/compare_bn_vs_nonparanormal.R
#
# =============================================================================

# Prevent OpenMP segfaults on macOS
Sys.setenv(OMP_NUM_THREADS = "1")

cat("=============================================================\n")
cat("  BN vs Nonparanormal Covariate Comparison\n")
cat("=============================================================\n\n")

# -- Source existing code -----------------------------------------------------
source("nonparanormal.R")
source("R/gaussian_copula_dag.R")
source("R/gaussian_copula_dag_sample.R")
source("R/rank_transform.R")

# -- Load libraries -----------------------------------------------------------
suppressPackageStartupMessages({
  library(VineCopula)
  library(tidyverse)
})

gcm_available <- requireNamespace("GeneralisedCovarianceMeasure", quietly = TRUE)
if (gcm_available) {
  library(GeneralisedCovarianceMeasure)
  cat("[INFO] GeneralisedCovarianceMeasure loaded.\n")
} else {
  cat("[WARN] GeneralisedCovarianceMeasure not available; GCM test skipped.\n")
}

# -- Ensure output directories exist ------------------------------------------
dir.create("results", showWarnings = FALSE)
dir.create("results/figures", showWarnings = FALSE)

# =============================================================================
# 1) Generate Dataset A (original Gamma BN chain)
# =============================================================================
cat("\n--- Step 1: Generating Dataset A (Gamma BN chain) ---\n")

N <- 50000
set.seed(42)

Z1_A <- rgamma(N, shape = 2, scale = 2)
Z2_A <- rgamma(N, shape = 1.5 * Z1_A + 2, scale = 2)
Z3_A <- rgamma(N, shape = 1.5 * Z2_A + 2, scale = 2)

Z_A <- cbind(Z1 = Z1_A, Z2 = Z2_A, Z3 = Z3_A)

cat(sprintf("  Dataset A: N = %d, dim = %d x %d\n", nrow(Z_A), nrow(Z_A), ncol(Z_A)))
cat(sprintf("  Z1: mean=%.2f, sd=%.2f\n", mean(Z1_A), sd(Z1_A)))
cat(sprintf("  Z2: mean=%.2f, sd=%.2f\n", mean(Z2_A), sd(Z2_A)))
cat(sprintf("  Z3: mean=%.2f, sd=%.2f\n", mean(Z3_A), sd(Z3_A)))

# =============================================================================
# 2) Fit latent Gaussian BN to Dataset A
# =============================================================================
cat("\n--- Step 2: Fitting latent Gaussian BN ---\n")

DAG_PARENTS <- list(
  Z1 = character(0),
  Z2 = "Z1",
  Z3 = "Z2"
)

fit <- fit_reference_gaussian_bn(Z_A, DAG_PARENTS)

cat("  Coefficient matrix B:\n")
print(round(fit$B, 4))
cat("\n  Innovation variances sigma2:\n")
print(round(fit$sigma2, 4))
cat("\n  Implied correlation matrix R:\n")
print(round(fit$R, 4))

# =============================================================================
# 3) Generate Dataset B (full nonparanormal surrogate)
# =============================================================================
cat("\n--- Step 3: Generating Dataset B (nonparanormal surrogate) ---\n")

# 3a) Simulate latent Gaussian chain via SEM
sim_B <- simulate_gaussian_bn_sem(N, fit, seed = 123)
Q_B <- sim_B$Q_tilde_Z   # latent Gaussian scores
U_B <- sim_B$tilde_U      # uniform marginals via Phi()

cat(sprintf("  Latent Gaussian: mean(Q1)=%.4f, sd(Q1)=%.4f\n",
            mean(Q_B[, 1]), sd(Q_B[, 1])))
cat(sprintf("  Uniform: mean(U1)=%.4f, range=(%.4f, %.4f)\n",
            mean(U_B[, 1]), min(U_B[, 1]), max(U_B[, 1])))

# 3b) Map uniforms to observed scale using empirical inverse CDFs from Dataset A
#     This ensures marginals match by construction.
cat("  Building empirical inverse CDFs from Dataset A...\n")

# For each variable, build an empirical quantile function from Dataset A
# Using sorted values and linearly interpolating
build_empirical_inv_cdf <- function(x) {
  x_sorted <- sort(x)
  n <- length(x_sorted)
  # Probabilities corresponding to sorted values: i/(n+1)
  probs <- seq_len(n) / (n + 1)
  # Use approxfun for linear interpolation, with rule=2 for clamping
  inv_cdf <- approxfun(probs, x_sorted, rule = 2)
  return(inv_cdf)
}

inv_cdf_Z1 <- build_empirical_inv_cdf(Z1_A)
inv_cdf_Z2 <- build_empirical_inv_cdf(Z2_A)
inv_cdf_Z3 <- build_empirical_inv_cdf(Z3_A)

Z1_B <- inv_cdf_Z1(U_B[, "Z1"])
Z2_B <- inv_cdf_Z2(U_B[, "Z2"])
Z3_B <- inv_cdf_Z3(U_B[, "Z3"])

Z_B <- cbind(Z1 = Z1_B, Z2 = Z2_B, Z3 = Z3_B)

cat(sprintf("  Dataset B: N = %d\n", nrow(Z_B)))
cat(sprintf("  Z1: mean=%.2f, sd=%.2f\n", mean(Z1_B), sd(Z1_B)))
cat(sprintf("  Z2: mean=%.2f, sd=%.2f\n", mean(Z2_B), sd(Z2_B)))
cat(sprintf("  Z3: mean=%.2f, sd=%.2f\n", mean(Z3_B), sd(Z3_B)))

# =============================================================================
# 4) Diagnostics: Compare conditional behaviour of A vs B
# =============================================================================
cat("\n--- Step 4: Running diagnostics ---\n")

# ---- 4a) Binned conditional mean and variance ----
cat("\n  4a) Binned conditional statistics (30 quantile bins)...\n")

compute_binned_stats <- function(parent, child, n_bins = 30) {
  # Create bins based on quantiles of parent
  breaks <- quantile(parent, probs = seq(0, 1, length.out = n_bins + 1))
  # Make breaks unique (can happen with discrete-ish data)
  breaks <- unique(breaks)
  n_bins_actual <- length(breaks) - 1

  bin_idx <- cut(parent, breaks = breaks, include.lowest = TRUE, labels = FALSE)

  results <- data.frame(
    bin = seq_len(n_bins_actual),
    parent_mid = numeric(n_bins_actual),
    cond_mean = numeric(n_bins_actual),
    cond_var = numeric(n_bins_actual),
    n_obs = integer(n_bins_actual)
  )

  for (b in seq_len(n_bins_actual)) {
    in_bin <- which(bin_idx == b)
    results$parent_mid[b] <- mean(parent[in_bin])
    results$cond_mean[b] <- mean(child[in_bin])
    results$cond_var[b] <- var(child[in_bin])
    results$n_obs[b] <- length(in_bin)
  }

  return(results)
}

# Edge Z1 -> Z2
stats_A_12 <- compute_binned_stats(Z1_A, Z2_A)
stats_A_12$dataset <- "A (Gamma BN)"
stats_A_12$edge <- "Z1 -> Z2"

stats_B_12 <- compute_binned_stats(Z1_B, Z2_B)
stats_B_12$dataset <- "B (Nonparanormal)"
stats_B_12$edge <- "Z1 -> Z2"

# Edge Z2 -> Z3
stats_A_23 <- compute_binned_stats(Z2_A, Z3_A)
stats_A_23$dataset <- "A (Gamma BN)"
stats_A_23$edge <- "Z2 -> Z3"

stats_B_23 <- compute_binned_stats(Z2_B, Z3_B)
stats_B_23$dataset <- "B (Nonparanormal)"
stats_B_23$edge <- "Z2 -> Z3"

all_stats <- bind_rows(stats_A_12, stats_B_12, stats_A_23, stats_B_23)

# Save CSV
write.csv(all_stats, "results/bn_vs_nonparanormal_binned_stats.csv", row.names = FALSE)
cat("  Saved binned stats to results/bn_vs_nonparanormal_binned_stats.csv\n")

# ---- 4b) Plots: conditional mean and variance curves ----
cat("\n  4b) Generating conditional mean/variance plots...\n")

# Conditional mean plot
p_mean <- ggplot(all_stats, aes(x = parent_mid, y = cond_mean,
                                  colour = dataset, linetype = dataset)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.5, alpha = 0.7) +
  facet_wrap(~ edge, scales = "free") +
  labs(
    title = "Conditional Mean: E[child | parent]",
    subtitle = "Gamma BN vs Nonparanormal Surrogate (30 quantile bins)",
    x = "Parent value (bin midpoint)",
    y = "E[child | parent]",
    colour = "Dataset", linetype = "Dataset"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

# Conditional variance plot
p_var <- ggplot(all_stats, aes(x = parent_mid, y = cond_var,
                                colour = dataset, linetype = dataset)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.5, alpha = 0.7) +
  facet_wrap(~ edge, scales = "free") +
  labs(
    title = "Conditional Variance: Var[child | parent]",
    subtitle = "Gamma BN vs Nonparanormal Surrogate (30 quantile bins)",
    x = "Parent value (bin midpoint)",
    y = "Var[child | parent]",
    colour = "Dataset", linetype = "Dataset"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

pdf("results/figures/bn_vs_nonparanormal_cond_mean.pdf", width = 10, height = 5)
print(p_mean)
dev.off()
cat("  Saved: results/figures/bn_vs_nonparanormal_cond_mean.pdf\n")

pdf("results/figures/bn_vs_nonparanormal_cond_var.pdf", width = 10, height = 5)
print(p_var)
dev.off()
cat("  Saved: results/figures/bn_vs_nonparanormal_cond_var.pdf\n")

# Combined plot (stacked)
p_combined <- gridExtra::grid.arrange(p_mean, p_var, ncol = 1)
pdf("results/figures/bn_vs_nonparanormal_combined.pdf", width = 10, height = 10)
gridExtra::grid.arrange(p_mean, p_var, ncol = 1)
dev.off()
cat("  Saved: results/figures/bn_vs_nonparanormal_combined.pdf\n")

# ---- 4c) Gamma GLM comparison ----
cat("\n  4c) Fitting Gamma GLMs (log link)...\n")

# Z1 -> Z2
glm_A_12 <- glm(Z2_A ~ Z1_A, family = Gamma(link = "log"))
glm_B_12 <- glm(Z2_B ~ Z1_B, family = Gamma(link = "log"))

# Z2 -> Z3
glm_A_23 <- glm(Z3_A ~ Z2_A, family = Gamma(link = "log"))
glm_B_23 <- glm(Z3_B ~ Z2_B, family = Gamma(link = "log"))

glm_summary <- data.frame(
  edge = c("Z1->Z2", "Z1->Z2", "Z2->Z3", "Z2->Z3"),
  dataset = c("A (Gamma BN)", "B (Nonparanormal)", "A (Gamma BN)", "B (Nonparanormal)"),
  deviance = c(deviance(glm_A_12), deviance(glm_B_12),
               deviance(glm_A_23), deviance(glm_B_23)),
  AIC = c(AIC(glm_A_12), AIC(glm_B_12),
          AIC(glm_A_23), AIC(glm_B_23)),
  intercept = c(coef(glm_A_12)[1], coef(glm_B_12)[1],
                coef(glm_A_23)[1], coef(glm_B_23)[1]),
  slope = c(coef(glm_A_12)[2], coef(glm_B_12)[2],
            coef(glm_A_23)[2], coef(glm_B_23)[2])
)

cat("\n  Gamma GLM summary:\n")
print(glm_summary, digits = 4, row.names = FALSE)

# ---- 4d) Rank correlations ----
cat("\n  4d) Rank correlations...\n")

cor_summary <- data.frame(
  edge = c("Z1-Z2", "Z1-Z2", "Z2-Z3", "Z2-Z3"),
  dataset = c("A", "B", "A", "B"),
  spearman = c(
    cor(Z1_A, Z2_A, method = "spearman"),
    cor(Z1_B, Z2_B, method = "spearman"),
    cor(Z2_A, Z3_A, method = "spearman"),
    cor(Z2_B, Z3_B, method = "spearman")
  ),
  kendall = c(
    cor(Z1_A, Z2_A, method = "kendall"),
    cor(Z1_B, Z2_B, method = "kendall"),
    cor(Z2_A, Z3_A, method = "kendall"),
    cor(Z2_B, Z3_B, method = "kendall")
  )
)

cat("\n  Rank correlation comparison:\n")
print(cor_summary, digits = 4, row.names = FALSE)

# ---- 4e) Markov CI test: Z1 _||_ Z3 | Z2 ----
cat("\n  4e) Markov conditional independence: Z1 _||_ Z3 | Z2...\n")

# Partial correlation on Gaussianised ranks
gaussianise <- function(x) {
  n <- length(x)
  qnorm(rank(x, ties.method = "average") / (n + 1))
}

G1_A <- gaussianise(Z1_A)
G2_A <- gaussianise(Z2_A)
G3_A <- gaussianise(Z3_A)

G1_B <- gaussianise(Z1_B)
G2_B <- gaussianise(Z2_B)
G3_B <- gaussianise(Z3_B)

# Partial correlation rho_{13|2}
partial_cor <- function(g1, g2, g3) {
  # Regress g1 on g2, regress g3 on g2, correlate residuals
  r1 <- residuals(lm(g1 ~ g2))
  r3 <- residuals(lm(g3 ~ g2))
  cor(r1, r3)
}

pcor_A <- partial_cor(G1_A, G2_A, G3_A)
pcor_B <- partial_cor(G1_B, G2_B, G3_B)

cat(sprintf("  Partial correlation rho_{Z1,Z3|Z2}:\n"))
cat(sprintf("    Dataset A (Gamma BN):       %.6f\n", pcor_A))
cat(sprintf("    Dataset B (Nonparanormal):  %.6f\n", pcor_B))

# Fisher z-test for partial correlation
fisher_z_test <- function(r, n, k = 1) {
  # k = number of conditioning variables
  z <- 0.5 * log((1 + r) / (1 - r))
  se <- 1 / sqrt(n - k - 3)
  p <- 2 * pnorm(-abs(z / se))
  list(z = z, se = se, statistic = z / se, p.value = p)
}

fz_A <- fisher_z_test(pcor_A, N)
fz_B <- fisher_z_test(pcor_B, N)

cat(sprintf("  Fisher z-test p-values:\n"))
cat(sprintf("    Dataset A: p = %.4e (z = %.4f)\n", fz_A$p.value, fz_A$statistic))
cat(sprintf("    Dataset B: p = %.4e (z = %.4f)\n", fz_B$p.value, fz_B$statistic))

# GCM test if available
if (gcm_available) {
  cat("\n  Running GCM test (N=5000 subsample for speed)...\n")
  set.seed(999)
  sub_idx <- sample(N, 5000)

  gcm_A <- tryCatch({
    gcm.test(Z1_A[sub_idx], Z3_A[sub_idx],
             as.data.frame(Z2_A[sub_idx]))
  }, error = function(e) {
    cat("    GCM error (A):", e$message, "\n")
    list(p.value = NA)
  })

  gcm_B <- tryCatch({
    gcm.test(Z1_B[sub_idx], Z3_B[sub_idx],
             as.data.frame(Z2_B[sub_idx]))
  }, error = function(e) {
    cat("    GCM error (B):", e$message, "\n")
    list(p.value = NA)
  })

  cat(sprintf("  GCM test p-values (Z1 _||_ Z3 | Z2):\n"))
  cat(sprintf("    Dataset A: p = %.4f\n", gcm_A$p.value))
  cat(sprintf("    Dataset B: p = %.4f\n", gcm_B$p.value))
}

# =============================================================================
# 5) Summary
# =============================================================================
cat("\n=============================================================\n")
cat("  SUMMARY\n")
cat("=============================================================\n\n")

cat("Dataset A = Original Gamma BN (heteroscedastic conditionals)\n")
cat("Dataset B = Nonparanormal surrogate (Gaussian copula + empirical margins)\n\n")

cat("Key observations:\n")
cat("  1. Marginals: By construction, marginals match exactly (empirical quantile map).\n")
cat("  2. Rank correlations: Spearman/Kendall should be very close (copula preserves rank dependence).\n")
cat("  3. Conditional mean: The Gamma BN has E[Z2|Z1] = (1.5*Z1+2)*2 = 3*Z1+4 (linear).\n")
cat("     The nonparanormal E[Z2|Z1] may differ in functional form.\n")
cat("  4. Conditional variance: The Gamma BN has Var[Z2|Z1] = (1.5*Z1+2)*4 (linear in Z1).\n")
cat("     The nonparanormal conditional variance will be different (likely not linear).\n")
cat("  5. Markov property: Z1 _||_ Z3 | Z2 holds exactly in both by DAG construction.\n\n")

cat("Gamma GLM fit comparison:\n")
print(glm_summary, digits = 4, row.names = FALSE)

cat("\nRank correlations:\n")
print(cor_summary, digits = 4, row.names = FALSE)

cat(sprintf("\nPartial correlations rho_{Z1,Z3|Z2}:\n"))
cat(sprintf("  A: %.6f  (p = %.2e)\n", pcor_A, fz_A$p.value))
cat(sprintf("  B: %.6f  (p = %.2e)\n", pcor_B, fz_B$p.value))

cat("\n  Output files:\n")
cat("    results/bn_vs_nonparanormal_binned_stats.csv\n")
cat("    results/figures/bn_vs_nonparanormal_cond_mean.pdf\n")
cat("    results/figures/bn_vs_nonparanormal_cond_var.pdf\n")
cat("    results/figures/bn_vs_nonparanormal_combined.pdf\n")

cat("\nDone.\n")
