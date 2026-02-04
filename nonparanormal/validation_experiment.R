#!/usr/bin/env Rscript
# =============================================================================
# Validation Experiment: Conditional Independence & Marginal Dependence
# =============================================================================
#
# This script validates the nonparanormal approach by testing:
# 1. CONDITIONAL INDEPENDENCE: Z1 ⊥ Z3 | Z2 (should be uniform p-values)
# 2. MARGINAL DEPENDENCE: Z1 NOT⊥ Z3 marginally (should be NON-uniform p-values)
#
# Model Structure (Gamma Bayesian Network):
#   Z1 ~ Gamma(2, 2)
#   Z2 | Z1 ~ Gamma(2 + 3*Z1, 1)
#   Z3 | Z2 ~ Gamma(2 + 1.5*Z2, 1)
#   Y | Z1, Z3 generated via nonparanormal copula
#
# =============================================================================

# Load required packages
suppressPackageStartupMessages({
  library(copula)
  library(VineCopula)
  library(ggplot2)
  library(gridExtra)
})

# Source the nonparanormal functions
source('nonparanormal.R')

# Set threading for macOS compatibility
Sys.setenv(OMP_NUM_THREADS = 1)

cat("\n")
cat("================================================================\n")
cat("  NONPARANORMAL VALIDATION EXPERIMENT\n")
cat("================================================================\n")
cat("\n")

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
set.seed(42)
sampleSize <- 20000  # Large sample for stable estimates
n_boot <- 100        # Number of bootstrap samples for CI tests
boot_sample_size <- 2000  # Size of each bootstrap sample

cat(sprintf("Sample size: %d\n", sampleSize))
cat(sprintf("Bootstrap replicates: %d\n", n_boot))
cat(sprintf("Bootstrap sample size: %d\n", boot_sample_size))
cat("\n")

# ------------------------------------------------------------------
# Step 1: Generate Data from Gamma Bayesian Network
# ------------------------------------------------------------------
cat("Step 1: Generating data from Gamma BN...\n")

# Z1 ~ Gamma(shape = 2, scale = 2)
U1 <- runif(sampleSize)
Z1 <- qgamma(U1, shape = 2, scale = 2)

# Z2 | Z1 ~ Gamma(shape = 2 + 3*Z1, scale = 1)
U2_1 <- runif(sampleSize)
Z2 <- qgamma(U2_1, shape = 2 + 3*Z1, scale = 1)

# Z3 | Z2 ~ Gamma(shape = 2 + 1.5*Z2, scale = 1)
U3_21 <- runif(sampleSize)
Z3 <- qgamma(U3_21, shape = 2 + 1.5*Z2, scale = 1)

cat(sprintf("  Z1: mean=%.2f, sd=%.2f\n", mean(Z1), sd(Z1)))
cat(sprintf("  Z2: mean=%.2f, sd=%.2f\n", mean(Z2), sd(Z2)))
cat(sprintf("  Z3: mean=%.2f, sd=%.2f\n", mean(Z3), sd(Z3)))

# Combine into data frames
covariate_data <- data.frame(Z1 = Z1, Z2 = Z2, Z3 = Z3)
cond_covariate_ranks <- data.frame(U1 = U1, U2_1 = U2_1, U3_21 = U3_21)

# ------------------------------------------------------------------
# Step 2: Generate Outcome via Nonparanormal Approximation
# ------------------------------------------------------------------
cat("\nStep 2: Generating outcome via nonparanormal copula...\n")

# Topological order and vine correlation parameters
topoOrder <- c(3, 1, 2)  # Z3 first, then Z1, then Z2
vine_cor_params <- c(0.5, 0.5, 0.0)  # rho(Y,Z3), rho(Y,Z1|Z3), rho(Y,Z2|Z1,Z3)

sampleResults <- simulateConditionalOutcomeSamples(
  covariate_data = covariate_data,
  cond_covariate_ranks = cond_covariate_ranks,
  vine_cor_params = vine_cor_params,
  topoOrder = topoOrder
)

cat("\nFitted Gaussian Copula Correlation Matrix (covariates only):\n")
print(round(sampleResults$gaussianCopulaFit$correlationMatrix, 3))

cat("\nFull Correlation Matrix (including Y):\n")
print(round(sampleResults$fullCorrelationMatrix, 3))

# Extract outcome
outcomeRankSamples <- as.vector(sampleResults$outcomeRankSamples)
Y <- qnorm(outcomeRankSamples)

cat(sprintf("\n  Y: mean=%.2f, sd=%.2f\n", mean(Y), sd(Y)))

# Scale all variables for testing
Z1_scaled <- scale(Z1)
Z2_scaled <- scale(Z2)
Z3_scaled <- scale(Z3)
Y_scaled <- scale(Y)

# ------------------------------------------------------------------
# Step 3: Verify Empirical Correlations
# ------------------------------------------------------------------
cat("\n")
cat("================================================================\n")
cat("  EMPIRICAL CORRELATIONS\n")
cat("================================================================\n")

full_data <- data.frame(Z1 = Z1, Z2 = Z2, Z3 = Z3, Y = Y)
emp_cor <- cor(full_data)
cat("\nEmpirical Correlation Matrix:\n")
print(round(emp_cor, 3))

# Calculate partial correlations
cat("\nKey Partial Correlations:\n")
# Z1 and Z3 given Z2 (should be ~0 due to Markov property)
resid_Z1_Z2 <- residuals(lm(Z1 ~ Z2))
resid_Z3_Z2 <- residuals(lm(Z3 ~ Z2))
pcor_Z1_Z3_given_Z2 <- cor(resid_Z1_Z2, resid_Z3_Z2)
cat(sprintf("  ρ(Z1, Z3 | Z2) = %.4f (should be ~0)\n", pcor_Z1_Z3_given_Z2))

# Z1 and Z3 marginally (should be non-zero)
cor_Z1_Z3 <- cor(Z1, Z3)
cat(sprintf("  ρ(Z1, Z3) marginal = %.4f (should be >0)\n", cor_Z1_Z3))

# Y and Z1 given Z3 (specified in vine)
resid_Y_Z3 <- residuals(lm(Y ~ Z3))
resid_Z1_Z3 <- residuals(lm(Z1 ~ Z3))
pcor_Y_Z1_given_Z3 <- cor(resid_Y_Z3, resid_Z1_Z3)
cat(sprintf("  ρ(Y, Z1 | Z3) = %.4f (should be ~0.5)\n", pcor_Y_Z1_given_Z3))

# ------------------------------------------------------------------
# Step 4: Conditional Independence Tests (GCM)
# ------------------------------------------------------------------
cat("\n")
cat("================================================================\n")
cat("  CONDITIONAL INDEPENDENCE TESTS (GCM)\n")
cat("================================================================\n")
cat("  These should show UNIFORM p-values (KS test p > 0.05)\n")
cat("================================================================\n\n")

# Check if GCM is available
if (!requireNamespace("GeneralisedCovarianceMeasure", quietly = TRUE)) {
  cat("WARNING: GeneralisedCovarianceMeasure not available. Using simplified Kendall test.\n")
  use_gcm <- FALSE
} else {
  library(GeneralisedCovarianceMeasure)
  use_gcm <- TRUE
}

# Test 1: Z1 ⊥ Z3 | Z2 (Markov property)
cat("Test 1: Z1 ⊥ Z3 | Z2 (Markov property)\n")
cat("  Running bootstrap...\n")
pvals_cond1 <- numeric(n_boot)
for (i in 1:n_boot) {
  idx <- sample(1:sampleSize, boot_sample_size, replace = TRUE)
  if (use_gcm) {
    test <- gcm.test(Z1_scaled[idx], Z3_scaled[idx], Z2_scaled[idx], regr.method = 'gam')
    pvals_cond1[i] <- test$p.value
  } else {
    # Fallback: partial correlation test
    resid1 <- residuals(lm(Z1_scaled[idx] ~ Z2_scaled[idx]))
    resid2 <- residuals(lm(Z3_scaled[idx] ~ Z2_scaled[idx]))
    pvals_cond1[i] <- cor.test(resid1, resid2)$p.value
  }
  if (i %% 20 == 0) cat(sprintf("    %d/%d\n", i, n_boot))
}
ks1 <- ks.test(pvals_cond1, "punif")
cat(sprintf("  KS test for uniformity: p = %.4f %s\n", ks1$p.value,
            ifelse(ks1$p.value > 0.05, "✓ PASS", "✗ FAIL")))

# Test 2: Z1 ⊥ Z3 | Z2, Y
cat("\nTest 2: Z1 ⊥ Z3 | Z2, Y\n")
cat("  Running bootstrap...\n")
pvals_cond2 <- numeric(n_boot)
for (i in 1:n_boot) {
  idx <- sample(1:sampleSize, boot_sample_size, replace = TRUE)
  Z <- cbind(Z2_scaled[idx], Y_scaled[idx])
  if (use_gcm) {
    test <- gcm.test(Z1_scaled[idx], Z3_scaled[idx], Z, regr.method = 'gam')
    pvals_cond2[i] <- test$p.value
  } else {
    resid1 <- residuals(lm(Z1_scaled[idx] ~ Z))
    resid2 <- residuals(lm(Z3_scaled[idx] ~ Z))
    pvals_cond2[i] <- cor.test(resid1, resid2)$p.value
  }
  if (i %% 20 == 0) cat(sprintf("    %d/%d\n", i, n_boot))
}
ks2 <- ks.test(pvals_cond2, "punif")
cat(sprintf("  KS test for uniformity: p = %.4f %s\n", ks2$p.value,
            ifelse(ks2$p.value > 0.05, "✓ PASS", "✗ FAIL")))

# Test 3: Z2 ⊥ Y | Z1, Z3 (should be independent by copula construction)
cat("\nTest 3: Z2 ⊥ Y | Z1, Z3\n")
cat("  Running bootstrap...\n")
pvals_cond3 <- numeric(n_boot)
for (i in 1:n_boot) {
  idx <- sample(1:sampleSize, boot_sample_size, replace = TRUE)
  Z <- cbind(Z1_scaled[idx], Z3_scaled[idx])
  if (use_gcm) {
    test <- gcm.test(Z2_scaled[idx], Y_scaled[idx], Z, regr.method = 'gam')
    pvals_cond3[i] <- test$p.value
  } else {
    resid1 <- residuals(lm(Z2_scaled[idx] ~ Z))
    resid2 <- residuals(lm(Y_scaled[idx] ~ Z))
    pvals_cond3[i] <- cor.test(resid1, resid2)$p.value
  }
  if (i %% 20 == 0) cat(sprintf("    %d/%d\n", i, n_boot))
}
ks3 <- ks.test(pvals_cond3, "punif")
cat(sprintf("  KS test for uniformity: p = %.4f %s\n", ks3$p.value,
            ifelse(ks3$p.value > 0.05, "✓ PASS", "✗ FAIL")))

# ------------------------------------------------------------------
# Step 5: Marginal Dependence Tests (Sanity Check)
# ------------------------------------------------------------------
cat("\n")
cat("================================================================\n")
cat("  MARGINAL DEPENDENCE TESTS (Sanity Check)\n")
cat("================================================================\n")
cat("  These should show NON-UNIFORM p-values (KS test p < 0.05)\n")
cat("================================================================\n\n")

# Test 4: Z1 NOT⊥ Z3 (marginally dependent via Z2)
cat("Test 4: Z1 NOT⊥ Z3 marginally (sanity check)\n")
cat("  Running bootstrap...\n")
pvals_marg1 <- numeric(n_boot)
for (i in 1:n_boot) {
  idx <- sample(1:sampleSize, boot_sample_size, replace = TRUE)
  # Using correlation test without conditioning
  pvals_marg1[i] <- cor.test(Z1_scaled[idx], Z3_scaled[idx])$p.value
  if (i %% 20 == 0) cat(sprintf("    %d/%d\n", i, n_boot))
}
ks4 <- ks.test(pvals_marg1, "punif")
cat(sprintf("  KS test for uniformity: p = %.6f %s\n", ks4$p.value,
            ifelse(ks4$p.value < 0.05, "✓ PASS (non-uniform)", "✗ FAIL")))
cat(sprintf("  Mean p-value: %.6f (should be very small)\n", mean(pvals_marg1)))

# Test 5: Z1 NOT⊥ Y (marginally dependent)
cat("\nTest 5: Z1 NOT⊥ Y marginally (sanity check)\n")
cat("  Running bootstrap...\n")
pvals_marg2 <- numeric(n_boot)
for (i in 1:n_boot) {
  idx <- sample(1:sampleSize, boot_sample_size, replace = TRUE)
  pvals_marg2[i] <- cor.test(Z1_scaled[idx], Y_scaled[idx])$p.value
  if (i %% 20 == 0) cat(sprintf("    %d/%d\n", i, n_boot))
}
ks5 <- ks.test(pvals_marg2, "punif")
cat(sprintf("  KS test for uniformity: p = %.6f %s\n", ks5$p.value,
            ifelse(ks5$p.value < 0.05, "✓ PASS (non-uniform)", "✗ FAIL")))
cat(sprintf("  Mean p-value: %.6f (should be very small)\n", mean(pvals_marg2)))

# ------------------------------------------------------------------
# Step 6: Summary
# ------------------------------------------------------------------
cat("\n")
cat("================================================================\n")
cat("  SUMMARY\n")
cat("================================================================\n\n")

results_df <- data.frame(
  Test = c("Z1 ⊥ Z3 | Z2", "Z1 ⊥ Z3 | Z2,Y", "Z2 ⊥ Y | Z1,Z3",
           "Z1 ⊥ Z3 (marginal)", "Z1 ⊥ Y (marginal)"),
  Expected = c("Independent", "Independent", "Independent",
               "Dependent", "Dependent"),
  KS_pvalue = round(c(ks1$p.value, ks2$p.value, ks3$p.value,
                      ks4$p.value, ks5$p.value), 4),
  Result = c(
    ifelse(ks1$p.value > 0.05, "PASS", "FAIL"),
    ifelse(ks2$p.value > 0.05, "PASS", "FAIL"),
    ifelse(ks3$p.value > 0.05, "PASS", "FAIL"),
    ifelse(ks4$p.value < 0.05, "PASS", "FAIL"),
    ifelse(ks5$p.value < 0.05, "PASS", "FAIL")
  )
)
print(results_df)

# Count passes
n_pass <- sum(results_df$Result == "PASS")
cat(sprintf("\n%d/%d tests passed\n", n_pass, nrow(results_df)))

# ------------------------------------------------------------------
# Step 7: Generate Plots
# ------------------------------------------------------------------
cat("\nGenerating plots...\n")

custom_theme <- theme_minimal() +
  theme(
    text = element_text(size = 12),
    axis.title = element_text(face = "bold"),
    panel.grid.major = element_line(color = "grey90"),
    panel.grid.minor = element_blank()
  )

# Conditional independence plots (should be uniform)
p1 <- ggplot(data.frame(p = pvals_cond1), aes(x = p)) +
  geom_histogram(bins = 10, fill = "steelblue", color = "black", alpha = 0.7) +
  geom_hline(yintercept = n_boot/10, color = "red", linetype = "dashed") +
  labs(title = "Z1 ⊥ Z3 | Z2", subtitle = sprintf("KS p = %.3f", ks1$p.value),
       x = "p-value", y = "Frequency") +
  custom_theme

p2 <- ggplot(data.frame(p = pvals_cond2), aes(x = p)) +
  geom_histogram(bins = 10, fill = "steelblue", color = "black", alpha = 0.7) +
  geom_hline(yintercept = n_boot/10, color = "red", linetype = "dashed") +
  labs(title = "Z1 ⊥ Z3 | Z2, Y", subtitle = sprintf("KS p = %.3f", ks2$p.value),
       x = "p-value", y = "Frequency") +
  custom_theme

p3 <- ggplot(data.frame(p = pvals_cond3), aes(x = p)) +
  geom_histogram(bins = 10, fill = "steelblue", color = "black", alpha = 0.7) +
  geom_hline(yintercept = n_boot/10, color = "red", linetype = "dashed") +
  labs(title = "Z2 ⊥ Y | Z1, Z3", subtitle = sprintf("KS p = %.3f", ks3$p.value),
       x = "p-value", y = "Frequency") +
  custom_theme

# Marginal dependence plots (should be non-uniform)
p4 <- ggplot(data.frame(p = pvals_marg1), aes(x = p)) +
  geom_histogram(bins = 10, fill = "coral", color = "black", alpha = 0.7) +
  geom_hline(yintercept = n_boot/10, color = "red", linetype = "dashed") +
  labs(title = "Z1 ⊥ Z3 (marginal)", subtitle = sprintf("KS p = %.2e", ks4$p.value),
       x = "p-value", y = "Frequency") +
  custom_theme

p5 <- ggplot(data.frame(p = pvals_marg2), aes(x = p)) +
  geom_histogram(bins = 10, fill = "coral", color = "black", alpha = 0.7) +
  geom_hline(yintercept = n_boot/10, color = "red", linetype = "dashed") +
  labs(title = "Z1 ⊥ Y (marginal)", subtitle = sprintf("KS p = %.2e", ks5$p.value),
       x = "p-value", y = "Frequency") +
  custom_theme

# Combine plots
combined_plot <- grid.arrange(
  p1, p2, p3, p4, p5,
  ncol = 3, nrow = 2,
  top = "Nonparanormal Validation: Blue = Conditional (should be uniform), Red = Marginal (should be non-uniform)"
)

# Save plots
if (!dir.exists("./results")) dir.create("./results")
ggsave("./results/validation_experiment.png", plot = combined_plot,
       width = 12, height = 8, dpi = 150)
cat("Plots saved to ./results/validation_experiment.png\n")

cat("\n")
cat("================================================================\n")
cat("  EXPERIMENT COMPLETE\n")
cat("================================================================\n")
