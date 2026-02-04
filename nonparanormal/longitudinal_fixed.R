#!/usr/bin/env Rscript
# =============================================================================
# Fixed Longitudinal Model - Fit Copula to Full Covariate History
# =============================================================================
#
# The fix: Instead of fitting the copula to only (Z1^t, Z2^t), we fit it to
# ALL covariates across ALL time points: (Z1^1, Z2^1, Z1^2, Z2^2, Z1^3, Z2^3)
#
# This allows uncondition_conditional_ranks to properly handle the temporal
# conditioning structure.
#
# =============================================================================

source('nonparanormal.R')
library(copula)
library(VineCopula)
library(GeneralisedCovarianceMeasure)

set.seed(42)
N <- 10000
T_max <- 3
n_boot <- 100
boot_sample_size <- 2000

cat('\n')
cat('================================================================\n')
cat('  FIXED LONGITUDINAL MODEL\n')
cat('  (Fitting copula to full covariate history)\n')
cat('================================================================\n\n')

# =============================================================================
# Step 1: Generate the full covariate history
# =============================================================================
cat('Step 1: Generating longitudinal covariates...\n')

# Storage for all covariates and ranks
Z_list <- list()
U_list <- list()

# Time 1 (roots)
U_list[["U1_1"]] <- runif(N)
Z_list[["Z1_1"]] <- qgamma(U_list[["U1_1"]], shape = 2, scale = 2)

U_list[["U2_1"]] <- runif(N)
Z_list[["Z2_1"]] <- qgamma(U_list[["U2_1"]], shape = 2 + 1.5 * Z_list[["Z1_1"]], scale = 1)

# Time 2, 3, ...
for (t in 2:T_max) {
  # Z1^t | Z1^{t-1}
  U_list[[paste0("U1_", t)]] <- runif(N)
  Z_list[[paste0("Z1_", t)]] <- qgamma(
    U_list[[paste0("U1_", t)]],
    shape = 2 + 1.5 * Z_list[[paste0("Z1_", t-1)]],
    scale = 1
  )

  # Z2^t | Z1^t, Z2^{t-1}
  U_list[[paste0("U2_", t)]] <- runif(N)
  Z_list[[paste0("Z2_", t)]] <- qgamma(
    U_list[[paste0("U2_", t)]],
    shape = 2 + Z_list[[paste0("Z1_", t)]] + 0.5 * Z_list[[paste0("Z2_", t-1)]],
    scale = 1
  )
}

# Print summary
for (t in 1:T_max) {
  cat(sprintf('  Time %d: Z1 mean=%.2f, Z2 mean=%.2f\n',
              t, mean(Z_list[[paste0("Z1_", t)]]), mean(Z_list[[paste0("Z2_", t)]])))
}

# =============================================================================
# Step 2: Build full covariate matrix and rank matrix
# =============================================================================
cat('\nStep 2: Building full covariate and rank matrices...\n')

# Order: Z1_1, Z2_1, Z1_2, Z2_2, Z1_3, Z2_3
cov_names <- c()
for (t in 1:T_max) {
  cov_names <- c(cov_names, paste0("Z1_", t), paste0("Z2_", t))
}

# Full covariate matrix
full_cov_data <- matrix(NA, nrow = N, ncol = 2 * T_max)
colnames(full_cov_data) <- cov_names
for (i in seq_along(cov_names)) {
  full_cov_data[, i] <- Z_list[[cov_names[i]]]
}

# Full conditional rank matrix (same order)
rank_names <- gsub("Z", "U", cov_names)
full_cond_ranks <- matrix(NA, nrow = N, ncol = 2 * T_max)
colnames(full_cond_ranks) <- rank_names
for (i in seq_along(rank_names)) {
  full_cond_ranks[, i] <- U_list[[rank_names[i]]]
}

cat(sprintf('  Full covariate matrix: %d x %d\n', nrow(full_cov_data), ncol(full_cov_data)))
cat(sprintf('  Columns: %s\n', paste(cov_names, collapse = ", ")))

# =============================================================================
# Step 3: Fit Gaussian copula to FULL covariate history
# =============================================================================
cat('\nStep 3: Fitting Gaussian copula to full history...\n')

gaussianCopulaFit <- fitMVGaussianCopula(full_cov_data, method = 'itau')
R_full <- gaussianCopulaFit$correlationMatrix

cat('\nFitted correlation matrix (6x6):\n')
colnames(R_full) <- cov_names
rownames(R_full) <- cov_names
print(round(R_full, 2))

# Check key partial correlations
cat('\nKey partial correlations in fitted copula:\n')
# Z1_2 vs Z1_1 | nothing (should be high)
cat(sprintf('  ρ(Z1_2, Z1_1) = %.3f (temporal AR)\n', R_full["Z1_2", "Z1_1"]))
# Z1_3 vs Z1_1 | Z1_2 (should be ~0 if Markov)
# Compute partial: ρ(Z1_3, Z1_1 | Z1_2)
r31 <- R_full["Z1_3", "Z1_1"]
r32 <- R_full["Z1_3", "Z1_2"]
r21 <- R_full["Z1_2", "Z1_1"]
partial_Z1 <- (r31 - r32 * r21) / sqrt((1 - r32^2) * (1 - r21^2))
cat(sprintf('  ρ(Z1_3, Z1_1 | Z1_2) = %.3f (should be ~0 if Markov)\n', partial_Z1))

# =============================================================================
# Step 4: Uncondition the full rank matrix WITH CORRECT DAG STRUCTURE
# =============================================================================
cat('\nStep 4: Unconditioning full rank matrix with DAG structure...\n')

# Use the helper function to generate the longitudinal DAG structure
# This creates the correct Markov conditioning structure:
#   Z1_1 -> Z2_1       (Z2_1 | Z1_1)
#   Z1_1 -> Z1_2       (Z1_2 | Z1_1, NOT Z2_1!)
#   Z1_2 -> Z2_2       (Z2_2 | Z1_2, Z2_1)
#   Z2_1 -> Z2_2
#   Z1_2 -> Z1_3       (Z1_3 | Z1_2)
#   Z1_3 -> Z2_3       (Z2_3 | Z1_3, Z2_2)
#   Z2_2 -> Z2_3
parents <- make_longitudinal_dag(n_time = T_max, n_cov = 2, structure = "markov")

cat('  DAG parent structure (using make_longitudinal_dag):\n')
dag_var_names <- attr(parents, "var_names")
for (j in 1:length(parents)) {
  pa_names <- if (length(parents[[j]]) == 0) "root" else paste(dag_var_names[parents[[j]]], collapse = ", ")
  cat(sprintf('    %s | %s\n', dag_var_names[j], pa_names))
}

marginal_ranks <- uncondition_conditional_ranks(full_cond_ranks, R_full, parents)
colnames(marginal_ranks) <- rank_names

cat('\n  Done. Checking marginal ranks are in [0,1]...\n')
cat(sprintf('  Range: [%.4f, %.4f]\n', min(marginal_ranks), max(marginal_ranks)))

# =============================================================================
# Step 5: Generate Y_t using ONLY current-time marginal ranks
# =============================================================================
cat('\nStep 5: Generating outcomes at each time point...\n')

# Vine correlation parameters for Y
rho_Y_Z2 <- 0.5
rho_Y_Z1_given_Z2 <- 0.8

Y_list <- list()

for (t in 1:T_max) {
  cat(sprintf('  Generating Y_%d...\n', t))

  # Get current time covariates
  Z1_t <- Z_list[[paste0("Z1_", t)]]
  Z2_t <- Z_list[[paste0("Z2_", t)]]

  # Get MARGINAL ranks for current time (from the full unconditioning)
  col_Z1 <- paste0("U1_", t)
  col_Z2 <- paste0("U2_", t)
  marginal_U1_t <- marginal_ranks[, col_Z1]
  marginal_U2_t <- marginal_ranks[, col_Z2]

  # Now use simulateMarginalOutcomeSamples (not Conditional!)
  # because we already have marginal ranks
  cov_data_t <- data.frame(Z1 = Z1_t, Z2 = Z2_t)

  # Fit copula to current time covariates for the Y generation
  copulaFit_t <- fitMVGaussianCopula(cov_data_t, method = 'itau')
  corMatrix_t <- copulaFit_t$correlationMatrix

  # Build full correlation matrix including Y
  topoOrder <- c(2, 1)
  vine_cor_params <- c(rho_Y_Z2, rho_Y_Z1_given_Z2)
  fullCorMatrix <- computeFullCorMatrix(topoOrder, corMatrix_t, vine_cor_params)

  # Generate Y using marginal ranks
  marginal_ranks_t <- cbind(marginal_U1_t, marginal_U2_t)
  X2_samples <- qnorm(marginal_ranks_t)

  outcome_model <- multivariate_conditional_mean_and_samples(
    X2_samples = X2_samples,
    R = fullCorMatrix
  )

  Y_t <- qnorm(pnorm(outcome_model$generated_samples))
  Y_list[[paste0("Y_", t)]] <- as.vector(Y_t)
}

# =============================================================================
# Step 6: Test Markov property
# =============================================================================
cat('\n')
cat('================================================================\n')
cat('  TESTING MARKOV PROPERTY\n')
cat('================================================================\n')

cat('\nPartial correlations (should be ~0 if Markov holds):\n')
for (t in 2:T_max) {
  Y_t <- Y_list[[paste0("Y_", t)]]
  Z1_t <- Z_list[[paste0("Z1_", t)]]
  Z2_t <- Z_list[[paste0("Z2_", t)]]
  Z1_prev <- Z_list[[paste0("Z1_", t-1)]]
  Z2_prev <- Z_list[[paste0("Z2_", t-1)]]

  # ρ(Y_t, Z1^{t-1} | Z^t)
  resid_Y <- residuals(lm(Y_t ~ Z1_t + Z2_t))
  resid_Z1_prev <- residuals(lm(Z1_prev ~ Z1_t + Z2_t))
  pcor1 <- cor(resid_Y, resid_Z1_prev)

  # ρ(Y_t, Z2^{t-1} | Z^t)
  resid_Z2_prev <- residuals(lm(Z2_prev ~ Z1_t + Z2_t))
  pcor2 <- cor(resid_Y, resid_Z2_prev)

  cat(sprintf('  Time %d: ρ(Y_%d, Z1^%d | Z^%d) = %.4f\n', t, t, t-1, t, pcor1))
  cat(sprintf('          ρ(Y_%d, Z2^%d | Z^%d) = %.4f\n', t, t-1, t, pcor2))
}

cat('\n')
cat('================================================================\n')
cat('  GCM CONDITIONAL INDEPENDENCE TESTS\n')
cat('================================================================\n')

results <- data.frame()

for (t in 2:T_max) {
  Y_t <- scale(Y_list[[paste0("Y_", t)]])
  Z1_t <- scale(Z_list[[paste0("Z1_", t)]])
  Z2_t <- scale(Z_list[[paste0("Z2_", t)]])
  Z1_prev <- scale(Z_list[[paste0("Z1_", t-1)]])
  Z2_prev <- scale(Z_list[[paste0("Z2_", t-1)]])

  # Test Y_t ⊥ Z1^{t-1} | Z^t
  cat(sprintf('\nTest: Y_%d ⊥ Z1^%d | Z^%d\n', t, t-1, t))
  pvals1 <- numeric(n_boot)
  for (i in 1:n_boot) {
    idx <- sample(1:N, boot_sample_size, replace = TRUE)
    test <- gcm.test(Y_t[idx], Z1_prev[idx], cbind(Z1_t[idx], Z2_t[idx]), regr.method = 'gam')
    pvals1[i] <- test$p.value
    if (i %% 25 == 0) cat(sprintf('  %d/%d\n', i, n_boot))
  }
  ks1 <- ks.test(pvals1, 'punif')
  cat(sprintf('  KS p-value: %.4f %s\n', ks1$p.value, ifelse(ks1$p.value > 0.05, '✓ PASS', '✗ FAIL')))

  results <- rbind(results, data.frame(
    Test = sprintf('Y_%d ⊥ Z1^%d | Z^%d', t, t-1, t),
    KS_pvalue = ks1$p.value,
    Pass = ks1$p.value > 0.05
  ))

  # Test Y_t ⊥ Z2^{t-1} | Z^t
  cat(sprintf('\nTest: Y_%d ⊥ Z2^%d | Z^%d\n', t, t-1, t))
  pvals2 <- numeric(n_boot)
  for (i in 1:n_boot) {
    idx <- sample(1:N, boot_sample_size, replace = TRUE)
    test <- gcm.test(Y_t[idx], Z2_prev[idx], cbind(Z1_t[idx], Z2_t[idx]), regr.method = 'gam')
    pvals2[i] <- test$p.value
    if (i %% 25 == 0) cat(sprintf('  %d/%d\n', i, n_boot))
  }
  ks2 <- ks.test(pvals2, 'punif')
  cat(sprintf('  KS p-value: %.4f %s\n', ks2$p.value, ifelse(ks2$p.value > 0.05, '✓ PASS', '✗ FAIL')))

  results <- rbind(results, data.frame(
    Test = sprintf('Y_%d ⊥ Z2^%d | Z^%d', t, t-1, t),
    KS_pvalue = ks2$p.value,
    Pass = ks2$p.value > 0.05
  ))
}

# =============================================================================
# Step 7: Sanity check - marginal dependence
# =============================================================================
cat('\n')
cat('================================================================\n')
cat('  MARGINAL DEPENDENCE (sanity check)\n')
cat('================================================================\n')

for (t in 2:T_max) {
  Y_t <- Y_list[[paste0("Y_", t)]]
  Z1_prev <- Z_list[[paste0("Z1_", t-1)]]

  cat(sprintf('\nTest: Y_%d ~ Z1^%d (marginal) - should be DEPENDENT\n', t, t-1))
  pvals <- numeric(n_boot)
  for (i in 1:n_boot) {
    idx <- sample(1:N, boot_sample_size, replace = TRUE)
    pvals[i] <- cor.test(Y_t[idx], Z1_prev[idx])$p.value
  }
  ks <- ks.test(pvals, 'punif')
  cat(sprintf('  KS p-value: %.4e %s\n', ks$p.value,
              ifelse(ks$p.value < 0.05, '✓ (dependent)', '✗ (unexpectedly independent)')))
}

# =============================================================================
# Summary
# =============================================================================
cat('\n')
cat('================================================================\n')
cat('  SUMMARY\n')
cat('================================================================\n\n')

print(results)

n_pass <- sum(results$Pass)
cat(sprintf('\n%d/%d Markov property tests passed\n', n_pass, nrow(results)))

if (n_pass == nrow(results)) {
  cat('\n✓ SUCCESS: Fitting copula to full history fixes the Markov property!\n')
} else {
  cat('\n✗ Some tests still failing - needs further investigation\n')
}
