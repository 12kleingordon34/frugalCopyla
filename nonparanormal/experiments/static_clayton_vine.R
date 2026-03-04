#' =============================================================================
#' Static Clayton Vine Experiment (Two-Simulator Design)
#' =============================================================================
#'
#' Tests whether the nonparanormal approximation preserves conditional
#' independence when evaluating a Clayton vine outcome copula. Two simulators
#' disentangle approximation error from intrinsic method behavior:
#'
#' - BN:    Real Gamma BN data + Gaussian projection (operational approximation)
#' - GAUSS: Data from the fitted Gaussian BN SEM (coherent baseline where
#'          Markov holds exactly)
#'
#' Both use the same Clayton vine (theta=2), the same CI tests, and produce
#' separate CSV + histogram outputs.
#'
#' Model specification:
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

# Prevent OpenMP segfaults in GCM/KCI on macOS
Sys.setenv(OMP_NUM_THREADS = "1")

library(VineCopula)
library(tidyverse)

# Source nonparanormal infrastructure
source("nonparanormal.R")
source("R/gaussian_copula_dag.R")
source("R/gaussian_copula_dag_sample.R")
source("R/rank_transform.R")

# Try to load GCM and KCI packages
gcm_available <- requireNamespace("GeneralisedCovarianceMeasure", quietly = TRUE)
if (gcm_available) library(GeneralisedCovarianceMeasure)

# KCI disabled -- too slow at large N (O(n^3)). Using RCoT instead.
kci_available <- FALSE
# kci_available <- requireNamespace("CondIndTests", quietly = TRUE)
# if (kci_available) library(CondIndTests)

# RCoT (fast nonparametric CI test via random Fourier features)
rcot_available <- requireNamespace("RCIT", quietly = TRUE) &&
                  requireNamespace("momentchi2", quietly = TRUE)
if (rcot_available) library(momentchi2)  # RCoT dependency, must be loaded

# Partial correlation fallback when GCM/KCI unavailable or fail
pcor_available <- requireNamespace("ppcor", quietly = TRUE)
if (pcor_available) library(ppcor)

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
  # BiCopHfunc2(u1, u2) = C(u1 | u2): first arg conditioned on second
  u_Z1_given_Z3 <- BiCopHfunc2(tilde_U_Z1, tilde_U_Z3,
                                 family = 1, par = rho_Z1_Z3)

  # Innovation for Y
  V <- runif(n)

  # Tree 2 inversion: V -> u_{Y|Z3}
  # BiCopHinv2(V, u2) inverts hfunc2: samples u1 given u2
  # Here: sample u_{Y|Z3} given u_{Z1|Z3}
  u_Y_given_Z3 <- BiCopHinv2(V, u_Z1_given_Z3,
                               family = 3, par = theta)

  # Tree 1 inversion: u_{Y|Z3} -> u_Y
  # BiCopHinv2(V, u2) inverts hfunc2: samples u1 given u2
  # Here: sample u_Y given u_{Z3}
  u_Y <- BiCopHinv2(u_Y_given_Z3, tilde_U_Z3,
                     family = 3, par = theta)

  return(list(
    u_Y = u_Y,
    u_Z1_given_Z3 = u_Z1_given_Z3,
    u_Y_given_Z3 = u_Y_given_Z3,
    V = V
  ))
}

# =============================================================================
# Shared Helpers
# =============================================================================

# Tiny helper: ifelse(is.na(p), NA, p < 0.05)
as0105 <- function(p) ifelse(is.na(p), NA, p < 0.05)

#' Run CI tests on four variables (Z1, Z2, Z3, Y)
#'
#' Null: Z2 _|_ Y | (Z1, Z3) -- should not reject
#' Alt:  Z1 _/|_ Z3 | (Y, Z2) -- should reject (collider)
run_ci_tests <- function(Z1, Z2, Z3, Y, rcot_seed = NULL) {
  cond_null <- cbind(Z1, Z3)
  cond_alt  <- cbind(Y, Z2)

  # GCM
  gcm_null_p <- gcm_alt_p <- NA
  if (gcm_available) {
    gcm_null_p <- tryCatch(
      gcm.test(X = Z2, Y = Y, Z = cond_null)$p.value,
      error = function(e) NA)
    gcm_alt_p <- tryCatch(
      gcm.test(X = Z1, Y = Z3, Z = cond_alt)$p.value,
      error = function(e) NA)
  }

  # RCoT (seed passed via seed= arg; external set.seed() is ineffective
  # because random_fourier_features() internally calls set.seed(seed))
  rcot_null_p <- rcot_alt_p <- NA
  if (rcot_available) {
    rcot_null_p <- tryCatch(
      RCIT::RCoT(Z2, Y, cond_null, seed = rcot_seed)$p,
      error = function(e) NA)
    rcot_alt_p <- tryCatch(
      RCIT::RCoT(Z1, Z3, cond_alt,
                 seed = if (!is.null(rcot_seed)) rcot_seed + 1L)$p,
      error = function(e) NA)
  }

  # Partial correlation
  pcor_null_p <- pcor_alt_p <- NA
  if (pcor_available) {
    pcor_null_p <- tryCatch(
      ppcor::pcor.test(Z2, Y, cbind(Z1, Z3))$p.value,
      error = function(e) NA)
    pcor_alt_p <- tryCatch(
      ppcor::pcor.test(Z1, Z3, cbind(Y, Z2))$p.value,
      error = function(e) NA)
  }

  # KCI (disabled)
  kci_null_p <- kci_alt_p <- NA

  data.frame(
    gcm_null_p = gcm_null_p, gcm_alt_p = gcm_alt_p,
    gcm_null_reject = as0105(gcm_null_p), gcm_alt_reject = as0105(gcm_alt_p),
    kci_null_p = kci_null_p, kci_alt_p = kci_alt_p,
    kci_null_reject = as0105(kci_null_p), kci_alt_reject = as0105(kci_alt_p),
    rcot_null_p = rcot_null_p, rcot_alt_p = rcot_alt_p,
    rcot_null_reject = as0105(rcot_null_p), rcot_alt_reject = as0105(rcot_alt_p),
    pcor_null_p = pcor_null_p, pcor_alt_p = pcor_alt_p,
    pcor_null_reject = as0105(pcor_null_p), pcor_alt_reject = as0105(pcor_alt_p),
    stringsAsFactors = FALSE
  )
}

#' Run diagnostics: KS uniformity + delta (BN only)
#'
#' @param tilde_U Marginal ranks matrix (n x 3)
#' @param u_Y Outcome rank vector
#' @param cond_ranks Conditional ranks matrix (n x 3), or NULL for GAUSS variant
run_diagnostics <- function(tilde_U, u_Y, cond_ranks = NULL) {
  ks_Z1_p <- ks.test(tilde_U[, 1], "punif")$p.value
  ks_Z2_p <- ks.test(tilde_U[, 2], "punif")$p.value
  ks_Z3_p <- ks.test(tilde_U[, 3], "punif")$p.value
  ks_Y_p  <- ks.test(u_Y, "punif")$p.value

  if (!is.null(cond_ranks)) {
    # BN variant: compute deltas
    delta_Z1 <- tilde_U[, 1] - cond_ranks[, 1]
    delta_Z2 <- tilde_U[, 2] - cond_ranks[, 2]
    delta_Z3 <- tilde_U[, 3] - cond_ranks[, 3]

    tail_Z2 <- cond_ranks[, 2] <= 0.05 | cond_ranks[, 2] >= 0.95
    tail_Z3 <- cond_ranks[, 3] <= 0.05 | cond_ranks[, 3] >= 0.95
    mid_Z2 <- !tail_Z2
    mid_Z3 <- !tail_Z3

    delta_Z1_mean_abs <- mean(abs(delta_Z1))
    delta_Z2_mean_abs <- mean(abs(delta_Z2))
    delta_Z3_mean_abs <- mean(abs(delta_Z3))
    delta_Z2_mean_abs_tail <- ifelse(sum(tail_Z2) > 0, mean(abs(delta_Z2[tail_Z2])), NA)
    delta_Z3_mean_abs_tail <- ifelse(sum(tail_Z3) > 0, mean(abs(delta_Z3[tail_Z3])), NA)
    delta_Z2_mean_abs_mid <- ifelse(sum(mid_Z2) > 0, mean(abs(delta_Z2[mid_Z2])), NA)
    delta_Z3_mean_abs_mid <- ifelse(sum(mid_Z3) > 0, mean(abs(delta_Z3[mid_Z3])), NA)
  } else {
    # GAUSS variant: no delta available
    delta_Z1_mean_abs <- delta_Z2_mean_abs <- delta_Z3_mean_abs <- NA
    delta_Z2_mean_abs_tail <- delta_Z3_mean_abs_tail <- NA
    delta_Z2_mean_abs_mid <- delta_Z3_mean_abs_mid <- NA
  }

  data.frame(
    ks_Z1_p = ks_Z1_p, ks_Z2_p = ks_Z2_p,
    ks_Z3_p = ks_Z3_p, ks_Y_p = ks_Y_p,
    delta_Z1_mean_abs = delta_Z1_mean_abs,
    delta_Z2_mean_abs = delta_Z2_mean_abs,
    delta_Z3_mean_abs = delta_Z3_mean_abs,
    delta_Z2_mean_abs_tail = delta_Z2_mean_abs_tail,
    delta_Z3_mean_abs_tail = delta_Z3_mean_abs_tail,
    delta_Z2_mean_abs_mid = delta_Z2_mean_abs_mid,
    delta_Z3_mean_abs_mid = delta_Z3_mean_abs_mid,
    stringsAsFactors = FALSE
  )
}

# =============================================================================
# Single Replication Functions
# =============================================================================

#' BN variant: real Gamma data + Gaussian projection
run_single_bn_simulation <- function(sim_id, n = N_SAMPLES) {
  bn_data <- generate_bn_covariates(n, seed = SEED_BASE + sim_id)

  tilde_U <- uncondition_conditional_ranks(
    cond_ranks = bn_data$cond_ranks,
    R = R_cov,
    parents = DAG_PARENTS
  )

  outcome <- sample_outcome_clayton_vine(
    tilde_U = tilde_U,
    theta = CLAYTON_THETA,
    rho_Z1_Z3 = rho_Z1_Z3,
    seed = SEED_BASE * 1000 + sim_id
  )

  u_Y <- outcome$u_Y
  Y <- qnorm(u_Y)

  # CI tests use observed Gamma Z
  ci <- run_ci_tests(bn_data$Z1, bn_data$Z2, bn_data$Z3, Y,
                     rcot_seed = SEED_BASE * 5000L + sim_id)
  diag_df <- run_diagnostics(tilde_U, u_Y, cond_ranks = bn_data$cond_ranks)

  cbind(
    data.frame(replication_id = sim_id, N = n, simulator = "BN",
               stringsAsFactors = FALSE),
    ci, diag_df
  )
}

#' GAUSS variant: data from the fitted Gaussian BN SEM
run_single_gauss_simulation <- function(sim_id, n = N_SAMPLES) {
  gauss <- simulate_gaussian_bn_sem(n, gaussian_bn_fit,
                                     seed = SEED_BASE + sim_id)
  tilde_U <- gauss$tilde_U

  outcome <- sample_outcome_clayton_vine(
    tilde_U = tilde_U,
    theta = CLAYTON_THETA,
    rho_Z1_Z3 = rho_Z1_Z3,
    seed = SEED_BASE * 1000 + sim_id
  )

  u_Y <- outcome$u_Y
  Y <- qnorm(u_Y)

  # CI tests use latent Gaussian scores (avoids boundary effects)
  ci <- run_ci_tests(gauss$Q_tilde_Z[, 1], gauss$Q_tilde_Z[, 2],
                     gauss$Q_tilde_Z[, 3], Y,
                     rcot_seed = SEED_BASE * 5000L + sim_id)
  diag_df <- run_diagnostics(tilde_U, u_Y, cond_ranks = NULL)

  cbind(
    data.frame(replication_id = sim_id, N = n, simulator = "GAUSS",
               stringsAsFactors = FALSE),
    ci, diag_df
  )
}

# =============================================================================
# Summary & Histogram Helpers
# =============================================================================

#' Print summary for one simulator variant
print_summary <- function(results_df, sim_type) {
  cat(sprintf("\n--- %s Simulator ---\n", sim_type))

  # GCM
  if (gcm_available && sum(!is.na(results_df$gcm_null_p)) > 0) {
    gcm_null_ps <- results_df$gcm_null_p[!is.na(results_df$gcm_null_p)]
    gcm_alt_ps <- results_df$gcm_alt_p[!is.na(results_df$gcm_alt_p)]
    ks_gcm_null <- ks.test(gcm_null_ps, "punif")
    gcm_alt_power <- mean(gcm_alt_ps < 0.05, na.rm = TRUE)
    cat("GCM CI Tests:\n")
    cat(sprintf("  Null (Z2 _|_ Y | Z1,Z3): KS p = %.4f\n", ks_gcm_null$p.value))
    cat(sprintf("  Alt  (Z1 _/|_ Z3 | Y,Z2): Power = %.4f\n", gcm_alt_power))
  } else {
    cat("GCM tests: not available\n")
  }

  # KCI
  if (kci_available && sum(!is.na(results_df$kci_null_p)) > 0) {
    kci_null_ps <- results_df$kci_null_p[!is.na(results_df$kci_null_p)]
    kci_alt_ps <- results_df$kci_alt_p[!is.na(results_df$kci_alt_p)]
    ks_kci_null <- ks.test(kci_null_ps, "punif")
    kci_alt_power <- mean(kci_alt_ps < 0.05, na.rm = TRUE)
    cat("KCI CI Tests:\n")
    cat(sprintf("  Null: KS p = %.4f\n", ks_kci_null$p.value))
    cat(sprintf("  Alt:  Power = %.4f\n", kci_alt_power))
  }

  # RCoT
  if (rcot_available && sum(!is.na(results_df$rcot_null_p)) > 0) {
    rcot_null_ps <- results_df$rcot_null_p[!is.na(results_df$rcot_null_p)]
    rcot_alt_ps <- results_df$rcot_alt_p[!is.na(results_df$rcot_alt_p)]
    ks_rcot_null <- ks.test(rcot_null_ps, "punif")
    rcot_alt_power <- mean(rcot_alt_ps < 0.05, na.rm = TRUE)
    cat("RCoT CI Tests:\n")
    cat(sprintf("  Null: KS p = %.4f\n", ks_rcot_null$p.value))
    cat(sprintf("  Alt:  Power = %.4f\n", rcot_alt_power))
  } else {
    cat("RCoT tests: not available\n")
  }

  # Partial correlation
  if (pcor_available && sum(!is.na(results_df$pcor_null_p)) > 0) {
    pcor_null_ps <- results_df$pcor_null_p[!is.na(results_df$pcor_null_p)]
    pcor_alt_ps <- results_df$pcor_alt_p[!is.na(results_df$pcor_alt_p)]
    ks_pcor_null <- ks.test(pcor_null_ps, "punif")
    pcor_alt_power <- mean(pcor_alt_ps < 0.05, na.rm = TRUE)
    cat("Partial Correlation Tests:\n")
    cat(sprintf("  Null: KS p = %.4f\n", ks_pcor_null$p.value))
    cat(sprintf("  Alt:  Power = %.4f\n", pcor_alt_power))
  } else {
    cat("Partial correlation tests: not available\n")
  }

  # Rank uniformity
  cat("Rank Uniformity (KS test, %% passing at alpha=0.05):\n")
  cat(sprintf("  tilde_U_Z1: %.1f%%\n", mean(results_df$ks_Z1_p > 0.05) * 100))
  cat(sprintf("  tilde_U_Z2: %.1f%%\n", mean(results_df$ks_Z2_p > 0.05) * 100))
  cat(sprintf("  tilde_U_Z3: %.1f%%\n", mean(results_df$ks_Z3_p > 0.05) * 100))
  cat(sprintf("  u_Y:        %.1f%%\n", mean(results_df$ks_Y_p > 0.05) * 100))

  # Delta diagnostics (BN only)
  if (sim_type == "BN" && !all(is.na(results_df$delta_Z1_mean_abs))) {
    cat("Delta Diagnostics (mean |Delta_d|):\n")
    cat(sprintf("  Z1 (root):  overall = %.4f\n",
                mean(results_df$delta_Z1_mean_abs)))
    cat(sprintf("  Z2:         overall = %.4f, tail = %.4f, mid = %.4f\n",
                mean(results_df$delta_Z2_mean_abs),
                mean(results_df$delta_Z2_mean_abs_tail, na.rm = TRUE),
                mean(results_df$delta_Z2_mean_abs_mid, na.rm = TRUE)))
    cat(sprintf("  Z3:         overall = %.4f, tail = %.4f, mid = %.4f\n",
                mean(results_df$delta_Z3_mean_abs),
                mean(results_df$delta_Z3_mean_abs_tail, na.rm = TRUE),
                mean(results_df$delta_Z3_mean_abs_mid, na.rm = TRUE)))
  }
}

#' Save p-value histogram for one simulator variant
save_pvalue_histogram <- function(results_df, sim_type) {
  pval_data <- data.frame()

  if (gcm_available && sum(!is.na(results_df$gcm_null_p)) > 0) {
    pval_data <- rbind(pval_data, data.frame(
      test = "GCM", hypothesis = "Null: Z2 _|_ Y | (Z1,Z3)",
      p_value = results_df$gcm_null_p[!is.na(results_df$gcm_null_p)]))
    pval_data <- rbind(pval_data, data.frame(
      test = "GCM", hypothesis = "Alt: Z1 _/|_ Z3 | (Y,Z2)",
      p_value = results_df$gcm_alt_p[!is.na(results_df$gcm_alt_p)]))
  }

  if (kci_available && sum(!is.na(results_df$kci_null_p)) > 0) {
    pval_data <- rbind(pval_data, data.frame(
      test = "KCI", hypothesis = "Null: Z2 _|_ Y | (Z1,Z3)",
      p_value = results_df$kci_null_p[!is.na(results_df$kci_null_p)]))
    pval_data <- rbind(pval_data, data.frame(
      test = "KCI", hypothesis = "Alt: Z1 _/|_ Z3 | (Y,Z2)",
      p_value = results_df$kci_alt_p[!is.na(results_df$kci_alt_p)]))
  }

  if (rcot_available && sum(!is.na(results_df$rcot_null_p)) > 0) {
    pval_data <- rbind(pval_data, data.frame(
      test = "RCoT", hypothesis = "Null: Z2 _|_ Y | (Z1,Z3)",
      p_value = results_df$rcot_null_p[!is.na(results_df$rcot_null_p)]))
    pval_data <- rbind(pval_data, data.frame(
      test = "RCoT", hypothesis = "Alt: Z1 _/|_ Z3 | (Y,Z2)",
      p_value = results_df$rcot_alt_p[!is.na(results_df$rcot_alt_p)]))
  }

  if (pcor_available && sum(!is.na(results_df$pcor_null_p)) > 0) {
    pval_data <- rbind(pval_data, data.frame(
      test = "Partial Cor", hypothesis = "Null: Z2 _|_ Y | (Z1,Z3)",
      p_value = results_df$pcor_null_p[!is.na(results_df$pcor_null_p)]))
    pval_data <- rbind(pval_data, data.frame(
      test = "Partial Cor", hypothesis = "Alt: Z1 _/|_ Z3 | (Y,Z2)",
      p_value = results_df$pcor_alt_p[!is.na(results_df$pcor_alt_p)]))
  }

  if (nrow(pval_data) > 0) {
    n_reps <- nrow(results_df)
    p_hist <- ggplot(pval_data, aes(x = p_value)) +
      geom_histogram(breaks = seq(0, 1, by = 0.05),
                     fill = "steelblue", colour = "white") +
      geom_hline(yintercept = n_reps * 0.05,
                 linetype = "dashed", colour = "red") +
      facet_grid(test ~ hypothesis, scales = "free_y") +
      labs(x = "p-value", y = "Count",
           title = sprintf("%s: CI Test p-values (N=%d, %d reps)",
                           sim_type, N_SAMPLES, N_SIMS),
           subtitle = "Null should be uniform; alternative near 0") +
      theme_minimal() +
      theme(strip.text = element_text(size = 9))

    fname <- sprintf("results/figures/static_clayton_pvals_%s.pdf", sim_type)
    ggsave(fname, p_hist, width = 8, height = 8)
    cat(sprintf("  Histogram saved to %s\n", fname))
  }
}

# =============================================================================
# Verification: Single Large Sample (Both Variants)
# =============================================================================

cat("=============================================================================\n")
cat("Verification: Single large sample (N=50000)\n")
cat("=============================================================================\n")

# --- BN verification ---
cat("\n--- BN variant ---\n")
set.seed(777)
verify_data <- generate_bn_covariates(50000, seed = 777)
verify_tilde_U <- uncondition_conditional_ranks(
  verify_data$cond_ranks, R_cov, parents = DAG_PARENTS
)
verify_outcome <- sample_outcome_clayton_vine(
  verify_tilde_U, CLAYTON_THETA, rho_Z1_Z3, seed = 778
)

fit_tree1 <- BiCopEst(verify_outcome$u_Y, verify_tilde_U[, 3], family = 3)
cat(sprintf("  Recovered Clayton theta (Tree 1, Y-Z3): %.3f (target: %.1f)\n",
            fit_tree1$par, CLAYTON_THETA))

verify_u_Y_given_Z3 <- BiCopHfunc2(
  verify_outcome$u_Y, verify_tilde_U[, 3], family = 3, par = CLAYTON_THETA
)
verify_u_Z1_given_Z3 <- BiCopHfunc2(
  verify_tilde_U[, 1], verify_tilde_U[, 3], family = 1, par = rho_Z1_Z3
)
fit_tree2 <- BiCopEst(verify_u_Y_given_Z3, verify_u_Z1_given_Z3, family = 3)
cat(sprintf("  Recovered Clayton theta (Tree 2, Y-Z1|Z3): %.3f (target: %.1f)\n",
            fit_tree2$par, CLAYTON_THETA))

ks_verify_Y <- ks.test(verify_outcome$u_Y, "punif")
cat(sprintf("  KS test for u_Y uniformity: p = %.4f\n", ks_verify_Y$p.value))

# --- GAUSS verification ---
cat("\n--- GAUSS variant ---\n")
gauss_verify <- simulate_gaussian_bn_sem(50000, gaussian_bn_fit, seed = 779)

# Sanity: empirical cor should match fit$R
cor_diff <- max(abs(cor(gauss_verify$Q_tilde_Z) - R_cov))
cat(sprintf("  GAUSS cor vs fit$R max diff: %.6f\n", cor_diff))

gauss_outcome <- sample_outcome_clayton_vine(
  gauss_verify$tilde_U, CLAYTON_THETA, rho_Z1_Z3, seed = 780
)

fit_tree1_g <- BiCopEst(gauss_outcome$u_Y, gauss_verify$tilde_U[, 3], family = 3)
cat(sprintf("  Recovered Clayton theta (Tree 1, Y-Z3): %.3f (target: %.1f)\n",
            fit_tree1_g$par, CLAYTON_THETA))

gauss_u_Y_given_Z3 <- BiCopHfunc2(
  gauss_outcome$u_Y, gauss_verify$tilde_U[, 3], family = 3, par = CLAYTON_THETA
)
gauss_u_Z1_given_Z3 <- BiCopHfunc2(
  gauss_verify$tilde_U[, 1], gauss_verify$tilde_U[, 3],
  family = 1, par = rho_Z1_Z3
)
fit_tree2_g <- BiCopEst(gauss_u_Y_given_Z3, gauss_u_Z1_given_Z3, family = 3)
cat(sprintf("  Recovered Clayton theta (Tree 2, Y-Z1|Z3): %.3f (target: %.1f)\n",
            fit_tree2_g$par, CLAYTON_THETA))

ks_verify_Y_g <- ks.test(gauss_outcome$u_Y, "punif")
cat(sprintf("  KS test for u_Y uniformity: p = %.4f\n", ks_verify_Y_g$p.value))

# KS for tilde_U columns
for (j in 1:3) {
  ks_j <- ks.test(gauss_verify$tilde_U[, j], "punif")
  cat(sprintf("  KS test for tilde_U[,%d] uniformity: p = %.4f\n", j, ks_j$p.value))
}

cat("\n")

# =============================================================================
# Run All Simulations (Both Variants)
# =============================================================================

if (!dir.exists("results")) dir.create("results", recursive = TRUE)
if (!dir.exists("results/figures")) dir.create("results/figures", recursive = TRUE)

for (sim_type in c("BN", "GAUSS")) {
  cat("=============================================================================\n")
  cat(sprintf("Running %s simulator (%d reps, N=%d)\n", sim_type, N_SIMS, N_SAMPLES))
  cat(sprintf("GCM: %s | RCoT: %s | pcor: %s\n",
              gcm_available, rcot_available, pcor_available))
  cat("=============================================================================\n")

  pb <- txtProgressBar(min = 0, max = N_SIMS, style = 3)
  results_list <- vector("list", N_SIMS)

  run_fn <- if (sim_type == "BN") run_single_bn_simulation else run_single_gauss_simulation

  for (i in 1:N_SIMS) {
    results_list[[i]] <- run_fn(i, n = N_SAMPLES)
    setTxtProgressBar(pb, i)
  }
  close(pb)

  results_df <- do.call(rbind, results_list)

  # Print summary
  print_summary(results_df, sim_type)

  # Save CSV
  csv_path <- sprintf("results/static_clayton_%s.csv", sim_type)
  write.csv(results_df, csv_path, row.names = FALSE)
  cat(sprintf("\n  Results saved to %s\n", csv_path))

  # Save histogram
  save_pvalue_histogram(results_df, sim_type)
}

# =============================================================================
# Verification Checklist (both variants)
# =============================================================================

cat("\n=============================================================================\n")
cat("VERIFICATION CHECKLIST\n")
cat("=============================================================================\n")

for (sim_type in c("BN", "GAUSS")) {
  csv_path <- sprintf("results/static_clayton_%s.csv", sim_type)
  if (!file.exists(csv_path)) next
  res <- read.csv(csv_path)

  cat(sprintf("\n--- %s ---\n", sim_type))

  if (gcm_available && sum(!is.na(res$gcm_null_p)) > 0) {
    gcm_null_ps <- res$gcm_null_p[!is.na(res$gcm_null_p)]
    gcm_alt_ps <- res$gcm_alt_p[!is.na(res$gcm_alt_p)]
    ks_p <- ks.test(gcm_null_ps, "punif")$p.value
    power <- mean(gcm_alt_ps < 0.05, na.rm = TRUE)
    cat(sprintf("[%s] GCM null uniform (KS p = %.3f > 0.05)\n",
                ifelse(ks_p > 0.05, "PASS", "WARN"), ks_p))
    cat(sprintf("[%s] GCM alt power > 0.8 (power = %.3f)\n",
                ifelse(power > 0.8, "PASS", "WARN"), power))
  }

  rank_pass_rate <- mean(c(
    res$ks_Z1_p > 0.05, res$ks_Z2_p > 0.05,
    res$ks_Z3_p > 0.05, res$ks_Y_p > 0.05
  ))
  cat(sprintf("[%s] Rank uniformity > 90%% (%.1f%%)\n",
              ifelse(rank_pass_rate > 0.9, "PASS", "WARN"),
              rank_pass_rate * 100))

  if (sim_type == "BN" && !all(is.na(res$delta_Z2_mean_abs_tail))) {
    tail_gt_mid_Z2 <- mean(res$delta_Z2_mean_abs_tail, na.rm = TRUE) >
                      mean(res$delta_Z2_mean_abs_mid, na.rm = TRUE)
    tail_gt_mid_Z3 <- mean(res$delta_Z3_mean_abs_tail, na.rm = TRUE) >
                      mean(res$delta_Z3_mean_abs_mid, na.rm = TRUE)
    cat(sprintf("[%s] Delta tails > mid for Z2: %.4f > %.4f\n",
                ifelse(tail_gt_mid_Z2, "PASS", "WARN"),
                mean(res$delta_Z2_mean_abs_tail, na.rm = TRUE),
                mean(res$delta_Z2_mean_abs_mid, na.rm = TRUE)))
    cat(sprintf("[%s] Delta tails > mid for Z3: %.4f > %.4f\n",
                ifelse(tail_gt_mid_Z3, "PASS", "WARN"),
                mean(res$delta_Z3_mean_abs_tail, na.rm = TRUE),
                mean(res$delta_Z3_mean_abs_mid, na.rm = TRUE)))
  }
}

cat("\n=============================================================================\n")

# =============================================================================
# Generate Summary CSVs for Paper
# =============================================================================

cat("\nGenerating summary CSVs...\n")

# --- CI Summary ---
ci_rows <- list()
for (sim_type in c("BN", "GAUSS")) {
  csv_path <- sprintf("results/static_clayton_%s.csv", sim_type)
  if (!file.exists(csv_path)) next
  res <- read.csv(csv_path)

  for (test_name in c("pcor", "gcm", "rcot")) {
    null_col <- paste0(test_name, "_null_p")
    alt_col  <- paste0(test_name, "_alt_p")
    if (!null_col %in% names(res)) next
    null_ps <- res[[null_col]][!is.na(res[[null_col]])]
    alt_ps  <- res[[alt_col]][!is.na(res[[alt_col]])]
    if (length(null_ps) == 0) next

    ks_null <- ks.test(null_ps, "punif")$p.value
    alt_reject <- mean(alt_ps < 0.05, na.rm = TRUE)
    null_reject <- mean(null_ps < 0.05, na.rm = TRUE)

    ci_rows[[length(ci_rows) + 1]] <- data.frame(
      simulator = sim_type, test = test_name,
      null_ks_p = round(ks_null, 3),
      null_reject_rate = round(null_reject, 3),
      alt_reject_rate = round(alt_reject, 3),
      stringsAsFactors = FALSE
    )
  }
}
ci_summary <- do.call(rbind, ci_rows)
write.csv(ci_summary, "results/static_ci_summary.csv", row.names = FALSE)
cat("  Saved results/static_ci_summary.csv\n")
print(ci_summary)

# --- Uniformity Summary ---
unif_rows <- list()
for (sim_type in c("BN", "GAUSS")) {
  csv_path <- sprintf("results/static_clayton_%s.csv", sim_type)
  if (!file.exists(csv_path)) next
  res <- read.csv(csv_path)

  for (var_info in list(
    list(col = "ks_Z1_p", var = "Z1"),
    list(col = "ks_Z2_p", var = "Z2"),
    list(col = "ks_Z3_p", var = "Z3"),
    list(col = "ks_Y_p",  var = "Y")
  )) {
    ps <- res[[var_info$col]]
    unif_rows[[length(unif_rows) + 1]] <- data.frame(
      simulator = sim_type, variable = var_info$var,
      pct_pass = round(mean(ps > 0.05) * 100, 1),
      mean_ks_p = round(mean(ps), 3),
      stringsAsFactors = FALSE
    )
  }
}
unif_summary <- do.call(rbind, unif_rows)
write.csv(unif_summary, "results/static_uniformity_summary.csv", row.names = FALSE)
cat("  Saved results/static_uniformity_summary.csv\n")
print(unif_summary)

# --- Self-Diagnosis ---
cat("\n=== SELF-DIAGNOSIS ===\n")
gauss_ci <- ci_summary[ci_summary$simulator == "GAUSS", ]
bn_ci <- ci_summary[ci_summary$simulator == "BN", ]

for (i in seq_len(nrow(gauss_ci))) {
  ok <- gauss_ci$null_ks_p[i] > 0.05
  cat(sprintf("[%s] GAUSS %s null KS p = %.3f (> 0.05)\n",
              ifelse(ok, "OK", "WARN"), gauss_ci$test[i], gauss_ci$null_ks_p[i]))
}

for (i in seq_len(nrow(ci_summary))) {
  if (ci_summary$test[i] == "rcot") {
    ok <- ci_summary$alt_reject_rate[i] > 0.05
    cat(sprintf("[%s] %s RCoT collider rejection = %.3f (> 0.05)\n",
                ifelse(ok, "OK", "WARN"), ci_summary$simulator[i],
                ci_summary$alt_reject_rate[i]))
  }
}

unif_gauss <- unif_summary[unif_summary$simulator == "GAUSS", ]
for (i in seq_len(nrow(unif_gauss))) {
  ok <- unif_gauss$pct_pass[i] >= 90 & unif_gauss$pct_pass[i] <= 99
  cat(sprintf("[%s] GAUSS %s uniformity pass rate = %.1f%%\n",
              ifelse(ok, "OK", "WARN"), unif_gauss$variable[i], unif_gauss$pct_pass[i]))
}

cat("=== END SELF-DIAGNOSIS ===\n")

# =============================================================================
# Generate LaTeX Table for Paper (CI Diagnostics)
# =============================================================================

cat("\nGenerating LaTeX table for CI diagnostics...\n")

tables_dir <- "../Hybrid-Frugal-Paper/tables"
if (!dir.exists(tables_dir)) dir.create(tables_dir, recursive = TRUE)

# Read the summary CSV we just wrote
ci_summary <- read.csv("results/static_ci_summary.csv", stringsAsFactors = FALSE)

# Test display names for the table
test_labels <- c(pcor = "pcor", gcm = "GCM", rcot = "RCoT")

# Build the LaTeX table matching the existing format in nonparanormal.tex
tex_lines <- c(
  "\\begin{table}[htbp]",
  "\\centering",
  paste0("\\caption{CI diagnostics for model $\\mathcal{M}_A$ (\\Cref{fig:MA-dag}), ",
         "based on 200 Monte Carlo replications with $N=1000$ each. ",
         "For the null relation $Z_2 \\indep Y \\mid (Z_1,Z_3)$ we report the KS $p$-value ",
         "for uniformity of test $p$-values. For the collider diagnostic ",
         "$Z_1 \\nindep Z_3 \\mid (Y,Z_2)$ we report the empirical rejection rate ",
         "at level $\\alpha=0.05$.}"),
  "\\label{tab:ci-tests-MA}",
  "\\begin{tabularx}{\\linewidth}{@{}p{0.44\\linewidth}p{0.24\\linewidth}cc@{}}",
  "\\toprule",
  "\\textbf{Case / Metric} & \\textbf{CI diagnostic} & \\textbf{Route A} & \\textbf{Route B} \\\\",
  "\\midrule"
)

# Null rows (Route A = GAUSS first, Route B = BN second)
for (i in seq_along(test_labels)) {
  test_name <- names(test_labels)[i]
  label <- test_labels[i]
  gauss_val <- ci_summary$null_ks_p[ci_summary$simulator == "GAUSS" & ci_summary$test == test_name]
  bn_val   <- ci_summary$null_ks_p[ci_summary$simulator == "BN"   & ci_summary$test == test_name]
  prefix <- if (i == 1) "\\textit{Null / KS $p$-value}" else ""
  tex_lines <- c(tex_lines, sprintf(
    "%s & %s & %.3f & %.3f \\\\", prefix, label, gauss_val, bn_val
  ))
}

tex_lines <- c(tex_lines, "\\addlinespace")

# Collider rows (Route A = GAUSS first, Route B = BN second)
for (i in seq_along(test_labels)) {
  test_name <- names(test_labels)[i]
  label <- test_labels[i]
  gauss_val <- ci_summary$alt_reject_rate[ci_summary$simulator == "GAUSS" & ci_summary$test == test_name]
  bn_val   <- ci_summary$alt_reject_rate[ci_summary$simulator == "BN"   & ci_summary$test == test_name]
  prefix <- if (i == 1) "\\textit{Collider / Rejection rate at $\\alpha=0.05$}" else ""
  tex_lines <- c(tex_lines, sprintf(
    "%s & %s & %.3f & %.3f \\\\", prefix, label, gauss_val, bn_val
  ))
}

tex_lines <- c(tex_lines,
  "\\bottomrule",
  "\\end{tabularx}",
  "\\end{table}"
)

tex_path <- file.path(tables_dir, "static_ci_diagnostics.tex")
writeLines(tex_lines, tex_path)
cat(sprintf("  Saved %s\n", tex_path))
