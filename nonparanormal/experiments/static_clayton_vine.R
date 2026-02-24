#' =============================================================================
#' Static Clayton Vine Experiment
#' =============================================================================
#'
#' Validates the nonparanormal approximation for a static model (M_B) with:
#' - BN covariate process: Z1 -> Z2 -> Z3 (conditional gamma densities)
#' - Outcome coupling: Clayton vine (theta=2) linking Y to {Z1, Z3}
#' - Gaussian h-functions for covariate conditioning edges
#' - Clayton h-functions for outcome edges
#'
#' DAG:  Z1 --> Z2 --> Z3 --> Y
#'       |                    ^
#'       +--------------------+
#'
#' Implied CI: Z2 _|_ Y | (Z1, Z3)
#' Collider:   Z1 _/|_ Z3 | (Y, Z2)
#'
#' =============================================================================

library(VineCopula)
library(tidyverse)

# Source nonparanormal infrastructure
source("nonparanormal.R")
source("R/gaussian_copula_dag.R")
source("R/rank_transform.R")

# Try to load GCM and KCI packages
gcm_available <- requireNamespace("GeneralisedCovarianceMeasure", quietly = TRUE)
if (gcm_available) library(GeneralisedCovarianceMeasure)

kci_available <- requireNamespace("CondIndTests", quietly = TRUE)
if (kci_available) library(CondIndTests)

# =============================================================================
# Parameters
# =============================================================================

N_SAMPLES <- 1000
N_SIMS <- 200
N_REF <- 50000        # Reference sample size for Gaussian BN fit
SEED_BASE <- 42
CLAYTON_THETA <- 2    # Clayton copula parameter for outcome edges

# BN specification: Z1 -> Z2 -> Z3 (chain)
DAG_PARENTS <- list(integer(0), c(1L), c(2L))

# =============================================================================
# BN Data Generation
# =============================================================================

#' Generate BN covariate data
#'
#' Z1 ~ Gamma(2, 2)
#' Z2 | Z1 ~ Gamma(1.5*Z1 + 2, 2)
#' Z3 | Z2 ~ Gamma(1.5*Z2 + 2, 2)
#'
#' @param n Sample size
#' @param seed Random seed
#' @return List with Z values and conditional ranks
generate_bn_covariates <- function(n, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  # Conditional ranks (independent Uniform)
  U_Z1 <- runif(n)
  U_Z2_given_Z1 <- runif(n)
  U_Z3_given_Z2 <- runif(n)

  # Transform through conditional CDFs
  Z1 <- qgamma(U_Z1, shape = 2, scale = 2)
  Z2 <- qgamma(U_Z2_given_Z1, shape = 1.5 * Z1 + 2, scale = 2)
  Z3 <- qgamma(U_Z3_given_Z2, shape = 1.5 * Z2 + 2, scale = 2)

  Z_mat <- cbind(Z1 = Z1, Z2 = Z2, Z3 = Z3)
  cond_ranks <- cbind(U_Z1, U_Z2_given_Z1, U_Z3_given_Z2)

  return(list(Z = Z_mat, cond_ranks = cond_ranks,
              Z1 = Z1, Z2 = Z2, Z3 = Z3))
}

# =============================================================================
# Fit Reference Gaussian BN (one-time calibration)
# =============================================================================

cat("Fitting reference Gaussian BN on N_ref =", N_REF, "samples...\n")
set.seed(999)
ref_data <- generate_bn_covariates(N_REF)
gaussian_bn_fit <- fit_reference_gaussian_bn(ref_data$Z, DAG_PARENTS)
R_cov <- gaussian_bn_fit$R

# Extract rho_{Z1,Z3} from the fitted correlation matrix
rho_Z1_Z3 <- R_cov[1, 3]
cat(sprintf("  R_cov:\n"))
print(round(R_cov, 4))
cat(sprintf("  rho(Z1, Z3) = %.4f\n\n", rho_Z1_Z3))

# =============================================================================
# Clayton Vine Outcome Sampling
# =============================================================================

#' Sample outcome Y from a Clayton vine coupled to covariates
#'
#' Vine structure (on copula parents {Z3, Z1}):
#'   Tree 1: Z3-Y edge (Clayton, theta)
#'   Tree 2: Z1-Y|Z3 edge (Clayton, theta)
#'
#' Covariate conditioning edge Z1|Z3 uses Gaussian h-function
#' (from the nonparanormal approximation).
#'
#' @param tilde_U Matrix of marginal covariate ranks (n x 3)
#' @param theta Clayton parameter
#' @param rho_Z1_Z3 Gaussian partial correlation for Z1-Z3 edge
#' @param seed Random seed
#' @return List with u_Y (outcome rank) and intermediate quantities
sample_outcome_clayton_vine <- function(tilde_U, theta, rho_Z1_Z3, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  n <- nrow(tilde_U)

  tilde_U_Z1 <- tilde_U[, 1]
  tilde_U_Z3 <- tilde_U[, 3]

  # Compute u_{Z1|Z3} via Gaussian h-function (covariate conditioning edge)
  u_Z1_given_Z3 <- BiCopHfunc1(tilde_U_Z1, tilde_U_Z3,
                                 family = 1, par = rho_Z1_Z3)

  # Innovation for Y
  V <- runif(n)

  # Tree 2 inversion: V -> u_{Y|Z3}
  # h_1(u_{Y|Z3}, u_{Z1|Z3}; Clayton, theta) = V
  # => u_{Y|Z3} = hinv_1(V, u_{Z1|Z3}; Clayton, theta)
  u_Y_given_Z3 <- BiCopHinv1(V, u_Z1_given_Z3,
                               family = 3, par = theta)

  # Tree 1 inversion: u_{Y|Z3} -> u_Y
  # h_1(u_Y, u_{Z3}; Clayton, theta) = u_{Y|Z3}
  # => u_Y = hinv_1(u_{Y|Z3}, u_{Z3}; Clayton, theta)
  u_Y <- BiCopHinv1(u_Y_given_Z3, tilde_U_Z3,
                     family = 3, par = theta)

  return(list(
    u_Y = u_Y,
    u_Z1_given_Z3 = u_Z1_given_Z3,
    u_Y_given_Z3 = u_Y_given_Z3,
    V = V
  ))
}

# =============================================================================
# Single Replication
# =============================================================================

run_single_static_simulation <- function(sim_id, n = N_SAMPLES, verbose = FALSE) {
  # Generate BN covariates
  bn_data <- generate_bn_covariates(n, seed = SEED_BASE + sim_id)

  # Uncondition conditional ranks to marginal ranks
  tilde_U <- uncondition_conditional_ranks(
    cond_ranks = bn_data$cond_ranks,
    R = R_cov,
    parents = DAG_PARENTS
  )

  # Sample outcome via Clayton vine
  outcome <- sample_outcome_clayton_vine(
    tilde_U = tilde_U,
    theta = CLAYTON_THETA,
    rho_Z1_Z3 = rho_Z1_Z3,
    seed = SEED_BASE * 1000 + sim_id
  )

  u_Y <- outcome$u_Y
  # Transform to observed scale (standard normal causal margin for simplicity)
  Y <- qnorm(u_Y)

  # ----- Diagnostics -----

  # 1. CI tests
  # Null: Z2 _|_ Y | (Z1, Z3) -- should not reject
  # Alt:  Z1 _/|_ Z3 | (Y, Z2) -- should reject (collider)
  cond_null <- cbind(bn_data$Z1, bn_data$Z3)
  cond_alt <- cbind(Y, bn_data$Z2)

  # GCM tests
  gcm_null_p <- NA
  gcm_alt_p <- NA
  if (gcm_available) {
    gcm_null <- tryCatch(
      gcm.test(X = bn_data$Z2, Y = Y, Z = cond_null),
      error = function(e) list(p.value = NA)
    )
    gcm_null_p <- gcm_null$p.value

    gcm_alt <- tryCatch(
      gcm.test(X = bn_data$Z1, Y = bn_data$Z3, Z = cond_alt),
      error = function(e) list(p.value = NA)
    )
    gcm_alt_p <- gcm_alt$p.value
  }

  # KCI tests
  kci_null_p <- NA
  kci_alt_p <- NA
  if (kci_available) {
    kci_null <- tryCatch(
      CondIndTests::KCI(bn_data$Z2, Y, cond_null,
                        GP = TRUE, width = 0, alpha = 0.05),
      error = function(e) list(pvalue = NA)
    )
    kci_null_p <- kci_null$pvalue

    kci_alt <- tryCatch(
      CondIndTests::KCI(bn_data$Z1, bn_data$Z3, cond_alt,
                        GP = TRUE, width = 0, alpha = 0.05),
      error = function(e) list(pvalue = NA)
    )
    kci_alt_p <- kci_alt$pvalue
  }

  # 2. Rank uniformity (KS tests)
  ks_Z1_p <- ks.test(tilde_U[, 1], "punif")$p.value
  ks_Z2_p <- ks.test(tilde_U[, 2], "punif")$p.value
  ks_Z3_p <- ks.test(tilde_U[, 3], "punif")$p.value
  ks_Y_p  <- ks.test(u_Y, "punif")$p.value

  # 3. Delta diagnostics: Delta_d = tilde_U_{Z_d} - U_{Z_d|pa(Z_d)}
  delta_Z1 <- tilde_U[, 1] - bn_data$cond_ranks[, 1]  # root: should be ~0
  delta_Z2 <- tilde_U[, 2] - bn_data$cond_ranks[, 2]
  delta_Z3 <- tilde_U[, 3] - bn_data$cond_ranks[, 3]

  # Tail indicators: U_{Z_d|pa(Z_d)} in [0, 0.05] or [0.95, 1]
  tail_Z2 <- bn_data$cond_ranks[, 2] <= 0.05 | bn_data$cond_ranks[, 2] >= 0.95
  tail_Z3 <- bn_data$cond_ranks[, 3] <= 0.05 | bn_data$cond_ranks[, 3] >= 0.95
  mid_Z2 <- !tail_Z2
  mid_Z3 <- !tail_Z3

  results <- data.frame(
    replication_id = sim_id,
    N = n,
    # GCM
    gcm_null_p = gcm_null_p,
    gcm_alt_p = gcm_alt_p,
    gcm_null_reject = ifelse(is.na(gcm_null_p), NA, gcm_null_p < 0.05),
    gcm_alt_reject = ifelse(is.na(gcm_alt_p), NA, gcm_alt_p < 0.05),
    # KCI
    kci_null_p = kci_null_p,
    kci_alt_p = kci_alt_p,
    kci_null_reject = ifelse(is.na(kci_null_p), NA, kci_null_p < 0.05),
    kci_alt_reject = ifelse(is.na(kci_alt_p), NA, kci_alt_p < 0.05),
    # Rank uniformity
    ks_Z1_p = ks_Z1_p,
    ks_Z2_p = ks_Z2_p,
    ks_Z3_p = ks_Z3_p,
    ks_Y_p = ks_Y_p,
    # Delta overall
    delta_Z1_mean_abs = mean(abs(delta_Z1)),
    delta_Z2_mean_abs = mean(abs(delta_Z2)),
    delta_Z3_mean_abs = mean(abs(delta_Z3)),
    # Delta tails
    delta_Z2_mean_abs_tail = ifelse(sum(tail_Z2) > 0, mean(abs(delta_Z2[tail_Z2])), NA),
    delta_Z3_mean_abs_tail = ifelse(sum(tail_Z3) > 0, mean(abs(delta_Z3[tail_Z3])), NA),
    # Delta middle
    delta_Z2_mean_abs_mid = ifelse(sum(mid_Z2) > 0, mean(abs(delta_Z2[mid_Z2])), NA),
    delta_Z3_mean_abs_mid = ifelse(sum(mid_Z3) > 0, mean(abs(delta_Z3[mid_Z3])), NA),
    stringsAsFactors = FALSE
  )

  if (verbose) {
    cat(sprintf("Sim %d: GCM null p=%.3f, alt p=%.3f | KS Y p=%.3f\n",
                sim_id, gcm_null_p, gcm_alt_p, ks_Y_p))
  }

  return(results)
}

# =============================================================================
# Verification: Single Large Sample
# =============================================================================

cat("=============================================================================\n")
cat("Verification: Single large sample (N=50000)\n")
cat("=============================================================================\n")

set.seed(777)
verify_data <- generate_bn_covariates(50000, seed = 777)
verify_tilde_U <- uncondition_conditional_ranks(
  verify_data$cond_ranks, R_cov, parents = DAG_PARENTS
)
verify_outcome <- sample_outcome_clayton_vine(
  verify_tilde_U, CLAYTON_THETA, rho_Z1_Z3, seed = 778
)

# Fit bivariate Clayton to (u_Y, tilde_U_Z3) - should recover theta ~ 2
fit_tree1 <- BiCopEst(verify_outcome$u_Y, verify_tilde_U[, 3], family = 3)
cat(sprintf("  Recovered Clayton theta (Tree 1, Y-Z3): %.3f (target: %.1f)\n",
            fit_tree1$par, CLAYTON_THETA))

# Fit bivariate Clayton to (u_{Y|Z3}, u_{Z1|Z3}) - should recover theta ~ 2
verify_u_Y_given_Z3 <- BiCopHfunc1(
  verify_outcome$u_Y, verify_tilde_U[, 3], family = 3, par = CLAYTON_THETA
)
verify_u_Z1_given_Z3 <- BiCopHfunc1(
  verify_tilde_U[, 1], verify_tilde_U[, 3], family = 1, par = rho_Z1_Z3
)
fit_tree2 <- BiCopEst(verify_u_Y_given_Z3, verify_u_Z1_given_Z3, family = 3)
cat(sprintf("  Recovered Clayton theta (Tree 2, Y-Z1|Z3): %.3f (target: %.1f)\n",
            fit_tree2$par, CLAYTON_THETA))

# Rank uniformity check
ks_verify_Y <- ks.test(verify_outcome$u_Y, "punif")
cat(sprintf("  KS test for u_Y uniformity: p = %.4f\n", ks_verify_Y$p.value))

cat("\n")

# =============================================================================
# Run All Simulations
# =============================================================================

cat("=============================================================================\n")
cat("Static Clayton Vine Experiment (Model M_B)\n")
cat("=============================================================================\n")
cat(sprintf("Sample size: %d\n", N_SAMPLES))
cat(sprintf("Number of replications: %d\n", N_SIMS))
cat(sprintf("Clayton theta: %.1f\n", CLAYTON_THETA))
cat(sprintf("GCM available: %s\n", gcm_available))
cat(sprintf("KCI available: %s\n", kci_available))
cat("=============================================================================\n\n")

cat("Running simulations...\n")
pb <- txtProgressBar(min = 0, max = N_SIMS, style = 3)

results_list <- vector("list", N_SIMS)
for (i in 1:N_SIMS) {
  results_list[[i]] <- run_single_static_simulation(i, verbose = FALSE)
  setTxtProgressBar(pb, i)
}
close(pb)

results_df <- do.call(rbind, results_list)

# =============================================================================
# Summary
# =============================================================================

cat("\n=============================================================================\n")
cat("RESULTS SUMMARY\n")
cat("=============================================================================\n\n")

# GCM diagnostics
if (gcm_available && sum(!is.na(results_df$gcm_null_p)) > 0) {
  gcm_null_ps <- results_df$gcm_null_p[!is.na(results_df$gcm_null_p)]
  gcm_alt_ps <- results_df$gcm_alt_p[!is.na(results_df$gcm_alt_p)]

  ks_gcm_null <- ks.test(gcm_null_ps, "punif")
  gcm_alt_power <- mean(gcm_alt_ps < 0.05, na.rm = TRUE)

  cat("GCM Conditional Independence Tests:\n")
  cat(sprintf("  Null (Z2 _|_ Y | Z1,Z3): KS p-value for uniformity = %.4f\n",
              ks_gcm_null$p.value))
  cat(sprintf("  Alt  (Z1 _/|_ Z3 | Y,Z2): Power (rejection rate) = %.4f\n",
              gcm_alt_power))
  cat("\n")
} else {
  cat("GCM tests: not available\n\n")
}

# KCI diagnostics
if (kci_available && sum(!is.na(results_df$kci_null_p)) > 0) {
  kci_null_ps <- results_df$kci_null_p[!is.na(results_df$kci_null_p)]
  kci_alt_ps <- results_df$kci_alt_p[!is.na(results_df$kci_alt_p)]

  ks_kci_null <- ks.test(kci_null_ps, "punif")
  kci_alt_power <- mean(kci_alt_ps < 0.05, na.rm = TRUE)

  cat("KCI Conditional Independence Tests:\n")
  cat(sprintf("  Null (Z2 _|_ Y | Z1,Z3): KS p-value for uniformity = %.4f\n",
              ks_kci_null$p.value))
  cat(sprintf("  Alt  (Z1 _/|_ Z3 | Y,Z2): Power (rejection rate) = %.4f\n",
              kci_alt_power))
  cat("\n")
}

# Rank uniformity
cat("Rank Uniformity (KS test, % passing at alpha=0.05):\n")
cat(sprintf("  tilde_U_Z1: %.1f%%\n", mean(results_df$ks_Z1_p > 0.05) * 100))
cat(sprintf("  tilde_U_Z2: %.1f%%\n", mean(results_df$ks_Z2_p > 0.05) * 100))
cat(sprintf("  tilde_U_Z3: %.1f%%\n", mean(results_df$ks_Z3_p > 0.05) * 100))
cat(sprintf("  u_Y:        %.1f%%\n", mean(results_df$ks_Y_p > 0.05) * 100))
cat("\n")

# Delta diagnostics
cat("Delta Diagnostics (mean |Delta_d|):\n")
cat(sprintf("  Z1 (root):  overall = %.4f\n", mean(results_df$delta_Z1_mean_abs)))
cat(sprintf("  Z2:         overall = %.4f, tail = %.4f, mid = %.4f\n",
            mean(results_df$delta_Z2_mean_abs),
            mean(results_df$delta_Z2_mean_abs_tail, na.rm = TRUE),
            mean(results_df$delta_Z2_mean_abs_mid, na.rm = TRUE)))
cat(sprintf("  Z3:         overall = %.4f, tail = %.4f, mid = %.4f\n",
            mean(results_df$delta_Z3_mean_abs),
            mean(results_df$delta_Z3_mean_abs_tail, na.rm = TRUE),
            mean(results_df$delta_Z3_mean_abs_mid, na.rm = TRUE)))
cat("\n")

# =============================================================================
# Save Results
# =============================================================================

if (!dir.exists("results")) dir.create("results", recursive = TRUE)

write.csv(results_df, "results/static_clayton_vine_results.csv", row.names = FALSE)
cat("Results saved to results/static_clayton_vine_results.csv\n")

# =============================================================================
# Verification Checklist
# =============================================================================

cat("\n=============================================================================\n")
cat("VERIFICATION CHECKLIST\n")
cat("=============================================================================\n")

if (gcm_available && sum(!is.na(results_df$gcm_null_p)) > 0) {
  cat(sprintf("[%s] GCM null p-values uniform (KS p = %.3f > 0.05)\n",
              ifelse(ks_gcm_null$p.value > 0.05, "PASS", "WARN"),
              ks_gcm_null$p.value))
  cat(sprintf("[%s] GCM alt power > 0.8 (power = %.3f)\n",
              ifelse(gcm_alt_power > 0.8, "PASS", "WARN"),
              gcm_alt_power))
}

rank_pass_rate <- mean(c(
  results_df$ks_Z1_p > 0.05,
  results_df$ks_Z2_p > 0.05,
  results_df$ks_Z3_p > 0.05,
  results_df$ks_Y_p > 0.05
))
cat(sprintf("[%s] Rank uniformity > 90%% (%.1f%%)\n",
            ifelse(rank_pass_rate > 0.9, "PASS", "WARN"),
            rank_pass_rate * 100))

# Delta tails > delta middle for non-root covariates
tail_gt_mid_Z2 <- mean(results_df$delta_Z2_mean_abs_tail, na.rm = TRUE) >
                  mean(results_df$delta_Z2_mean_abs_mid, na.rm = TRUE)
tail_gt_mid_Z3 <- mean(results_df$delta_Z3_mean_abs_tail, na.rm = TRUE) >
                  mean(results_df$delta_Z3_mean_abs_mid, na.rm = TRUE)
cat(sprintf("[%s] Delta tails > mid for Z2: %.4f > %.4f\n",
            ifelse(tail_gt_mid_Z2, "PASS", "WARN"),
            mean(results_df$delta_Z2_mean_abs_tail, na.rm = TRUE),
            mean(results_df$delta_Z2_mean_abs_mid, na.rm = TRUE)))
cat(sprintf("[%s] Delta tails > mid for Z3: %.4f > %.4f\n",
            ifelse(tail_gt_mid_Z3, "PASS", "WARN"),
            mean(results_df$delta_Z3_mean_abs_tail, na.rm = TRUE),
            mean(results_df$delta_Z3_mean_abs_mid, na.rm = TRUE)))

cat("=============================================================================\n")
