#' =============================================================================
#' Route A vs Route B: Static CI Diagnostics
#' =============================================================================
#'
#' Compares two implementation routes for the nonparanormal approximation:
#'
#' - Route A (GAUSS): Fully coherent Gaussian SEM simulator -- all ranks exact
#' - Route B (BN):    BN-preserving rank construction via
#'                    uncondition_conditional_ranks()
#'
#' Both routes use the same Clayton vine outcome copula (theta=2) and the
#' same CI tests (pcor, RCoT, GCM).
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

# Try to load CI test packages
gcm_available <- requireNamespace("GeneralisedCovarianceMeasure", quietly = TRUE)
if (gcm_available) library(GeneralisedCovarianceMeasure)

rcot_available <- requireNamespace("RCIT", quietly = TRUE) &&
                  requireNamespace("momentchi2", quietly = TRUE)
if (rcot_available) library(momentchi2)

pcor_available <- requireNamespace("ppcor", quietly = TRUE)
if (pcor_available) library(ppcor)

# =============================================================================
# Parameters
# =============================================================================

N_SAMPLES <- 1000
N_SIMS <- 200
N_REF <- 50000
SEED_BASE <- 42
CLAYTON_THETA <- 2

# BN specification: Z1 -> Z2 -> Z3 (chain)
DAG_PARENTS <- list(integer(0), c(1L), c(2L))

# =============================================================================
# BN Data Generation (copied from static_clayton_vine.R)
# =============================================================================

generate_bn_covariates <- function(n, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  U_Z1 <- runif(n)
  U_Z2_given_Z1 <- runif(n)
  U_Z3_given_Z2 <- runif(n)

  Z1 <- qgamma(U_Z1, shape = 2, scale = 2)
  Z2 <- qgamma(U_Z2_given_Z1, shape = 1.5 * Z1 + 2, scale = 2)
  Z3 <- qgamma(U_Z3_given_Z2, shape = 1.5 * Z2 + 2, scale = 2)

  Z_mat <- cbind(Z1 = Z1, Z2 = Z2, Z3 = Z3)
  cond_ranks <- cbind(U_Z1, U_Z2_given_Z1, U_Z3_given_Z2)

  list(Z = Z_mat, cond_ranks = cond_ranks,
       Z1 = Z1, Z2 = Z2, Z3 = Z3)
}

# =============================================================================
# Clayton Vine Outcome Sampling (copied from static_clayton_vine.R)
# =============================================================================

sample_outcome_clayton_vine <- function(tilde_U, theta, rho_Z1_Z3, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  n <- nrow(tilde_U)

  tilde_U_Z1 <- tilde_U[, 1]
  tilde_U_Z3 <- tilde_U[, 3]

  # Compute u_{Z1|Z3} via Gaussian h-function
  u_Z1_given_Z3 <- BiCopHfunc2(tilde_U_Z1, tilde_U_Z3,
                                 family = 1, par = rho_Z1_Z3)

  V <- runif(n)

  # Tree 2 inversion: V -> u_{Y|Z3}
  u_Y_given_Z3 <- BiCopHinv2(V, u_Z1_given_Z3,
                               family = 3, par = theta)

  # Tree 1 inversion: u_{Y|Z3} -> u_Y
  u_Y <- BiCopHinv2(u_Y_given_Z3, tilde_U_Z3,
                     family = 3, par = theta)

  list(u_Y = u_Y, u_Z1_given_Z3 = u_Z1_given_Z3,
       u_Y_given_Z3 = u_Y_given_Z3, V = V)
}

# =============================================================================
# Fit Reference Gaussian BN (one-time calibration)
# =============================================================================

cat("Fitting reference Gaussian BN on N_ref =", N_REF, "samples...\n")
set.seed(999)
ref_data <- generate_bn_covariates(N_REF)
gaussian_bn_fit <- fit_reference_gaussian_bn(ref_data$Z, DAG_PARENTS)
R_cov <- gaussian_bn_fit$R

rho_Z1_Z3 <- R_cov[1, 3]
cat(sprintf("  R_cov:\n"))
print(round(R_cov, 4))
cat(sprintf("  rho(Z1, Z3) = %.4f\n\n", rho_Z1_Z3))

# =============================================================================
# Shared Helpers
# =============================================================================

as0105 <- function(p) ifelse(is.na(p), NA, p < 0.05)

run_ci_tests <- function(Z1, Z2, Z3, Y, rcot_seed = NULL) {
  cond_null <- cbind(Z1, Z3)
  cond_alt  <- cbind(Y, Z2)

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

  pcor_null_p <- pcor_alt_p <- NA
  if (pcor_available) {
    pcor_null_p <- tryCatch(
      ppcor::pcor.test(Z2, Y, cbind(Z1, Z3))$p.value,
      error = function(e) NA)
    pcor_alt_p <- tryCatch(
      ppcor::pcor.test(Z1, Z3, cbind(Y, Z2))$p.value,
      error = function(e) NA)
  }

  data.frame(
    gcm_null_p = gcm_null_p, gcm_alt_p = gcm_alt_p,
    gcm_null_reject = as0105(gcm_null_p), gcm_alt_reject = as0105(gcm_alt_p),
    rcot_null_p = rcot_null_p, rcot_alt_p = rcot_alt_p,
    rcot_null_reject = as0105(rcot_null_p), rcot_alt_reject = as0105(rcot_alt_p),
    pcor_null_p = pcor_null_p, pcor_alt_p = pcor_alt_p,
    pcor_null_reject = as0105(pcor_null_p), pcor_alt_reject = as0105(pcor_alt_p),
    stringsAsFactors = FALSE
  )
}

# =============================================================================
# Single Replication Functions
# =============================================================================

#' Route A: Gaussian SEM simulator (GAUSS baseline)
run_single_routeA <- function(sim_id, n = N_SAMPLES) {
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

  # CI tests on Gaussian scores (avoids boundary effects)
  ci <- run_ci_tests(gauss$Q_tilde_Z[, 1], gauss$Q_tilde_Z[, 2],
                     gauss$Q_tilde_Z[, 3], Y,
                     rcot_seed = SEED_BASE * 5000L + sim_id)

  cbind(
    data.frame(replication_id = sim_id, route = "A",
               stringsAsFactors = FALSE),
    ci
  )
}

#' Route B: BN-preserving rank construction
run_single_routeB <- function(sim_id, n = N_SAMPLES) {
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

  # CI tests on observed Gamma Z values
  ci <- run_ci_tests(bn_data$Z1, bn_data$Z2, bn_data$Z3, Y,
                     rcot_seed = SEED_BASE * 5000L + sim_id)

  cbind(
    data.frame(replication_id = sim_id, route = "B",
               stringsAsFactors = FALSE),
    ci
  )
}

# =============================================================================
# Run All Simulations
# =============================================================================

if (!dir.exists("results")) dir.create("results", recursive = TRUE)
if (!dir.exists("results/figures")) dir.create("results/figures", recursive = TRUE)

all_results <- list()

for (route in c("A", "B")) {
  cat("=============================================================================\n")
  cat(sprintf("Running Route %s (%d reps, N=%d)\n", route, N_SIMS, N_SAMPLES))
  cat(sprintf("GCM: %s | RCoT: %s | pcor: %s\n",
              gcm_available, rcot_available, pcor_available))
  cat("=============================================================================\n")

  pb <- txtProgressBar(min = 0, max = N_SIMS, style = 3)
  results_list <- vector("list", N_SIMS)

  run_fn <- if (route == "A") run_single_routeA else run_single_routeB

  for (i in 1:N_SIMS) {
    results_list[[i]] <- run_fn(i, n = N_SAMPLES)
    setTxtProgressBar(pb, i)
  }
  close(pb)

  all_results[[route]] <- do.call(rbind, results_list)
}

results_df <- do.call(rbind, all_results)

# =============================================================================
# Save Per-Rep CSV
# =============================================================================

write.csv(results_df, "results/static_routeA_vs_routeB_ci.csv", row.names = FALSE)
cat("\nSaved results/static_routeA_vs_routeB_ci.csv\n")

# =============================================================================
# Compute Summary
# =============================================================================

ci_rows <- list()
for (route in c("A", "B")) {
  res <- results_df[results_df$route == route, ]

  for (test_name in c("pcor", "gcm", "rcot")) {
    null_col <- paste0(test_name, "_null_p")
    alt_col  <- paste0(test_name, "_alt_p")
    if (!null_col %in% names(res)) next
    null_ps <- res[[null_col]][!is.na(res[[null_col]])]
    alt_ps  <- res[[alt_col]][!is.na(res[[alt_col]])]
    if (length(null_ps) == 0) next

    ks_null <- ks.test(null_ps, "punif")$p.value
    null_reject <- mean(null_ps < 0.05, na.rm = TRUE)
    alt_reject <- mean(alt_ps < 0.05, na.rm = TRUE)

    ci_rows[[length(ci_rows) + 1]] <- data.frame(
      route = route, test = test_name,
      null_ks_p = round(ks_null, 3),
      null_reject_rate = round(null_reject, 3),
      alt_reject_rate = round(alt_reject, 3),
      stringsAsFactors = FALSE
    )
  }
}

ci_summary <- do.call(rbind, ci_rows)
write.csv(ci_summary, "results/static_routeA_vs_routeB_ci_summary.csv", row.names = FALSE)
cat("Saved results/static_routeA_vs_routeB_ci_summary.csv\n")
print(ci_summary)

# =============================================================================
# P-value Histogram
# =============================================================================

pval_data <- data.frame()

for (route in c("A", "B")) {
  res <- results_df[results_df$route == route, ]
  route_label <- ifelse(route == "A", "Route A (GAUSS)", "Route B (BN)")

  for (test_info in list(
    list(name = "Partial Cor", null_col = "pcor_null_p", alt_col = "pcor_alt_p"),
    list(name = "RCoT", null_col = "rcot_null_p", alt_col = "rcot_alt_p"),
    list(name = "GCM", null_col = "gcm_null_p", alt_col = "gcm_alt_p")
  )) {
    null_ps <- res[[test_info$null_col]][!is.na(res[[test_info$null_col]])]
    alt_ps  <- res[[test_info$alt_col]][!is.na(res[[test_info$alt_col]])]

    if (length(null_ps) > 0) {
      pval_data <- rbind(pval_data, data.frame(
        test = test_info$name, hypothesis = "Null: Z2 _|_ Y | (Z1,Z3)",
        route = route_label, p_value = null_ps))
    }
    if (length(alt_ps) > 0) {
      pval_data <- rbind(pval_data, data.frame(
        test = test_info$name, hypothesis = "Alt: Z1 _/|_ Z3 | (Y,Z2)",
        route = route_label, p_value = alt_ps))
    }
  }
}

if (nrow(pval_data) > 0) {
  n_reps <- N_SIMS
  p_hist <- ggplot(pval_data, aes(x = p_value, fill = route)) +
    geom_histogram(breaks = seq(0, 1, by = 0.05),
                   position = "dodge", colour = "white", alpha = 0.8) +
    geom_hline(yintercept = n_reps * 0.05,
               linetype = "dashed", colour = "red") +
    facet_grid(test ~ hypothesis, scales = "free_y") +
    scale_fill_manual(values = c("Route A (GAUSS)" = "steelblue",
                                 "Route B (BN)" = "coral")) +
    labs(x = "p-value", y = "Count",
         title = sprintf("Route A vs Route B: CI Test p-values (N=%d, %d reps)",
                         N_SAMPLES, N_SIMS),
         subtitle = "Null should be uniform; alternative near 0",
         fill = "Route") +
    theme_minimal() +
    theme(strip.text = element_text(size = 9),
          legend.position = "bottom")

  ggsave("results/figures/static_routeA_vs_routeB_pval_hist.pdf",
         p_hist, width = 9, height = 8)
  cat("Saved results/figures/static_routeA_vs_routeB_pval_hist.pdf\n")
}

# =============================================================================
# Self-Diagnosis
# =============================================================================

cat("\n=== SELF-DIAGNOSIS ===\n")

for (route in c("A", "B")) {
  route_summary <- ci_summary[ci_summary$route == route, ]
  cat(sprintf("\n--- Route %s ---\n", route))

  for (i in seq_len(nrow(route_summary))) {
    test <- route_summary$test[i]
    ks_p <- route_summary$null_ks_p[i]
    null_rej <- route_summary$null_reject_rate[i]
    alt_rej <- route_summary$alt_reject_rate[i]

    ok_null <- ks_p > 0.05
    ok_null_rate <- null_rej <= 0.10
    ok_alt <- alt_rej > 0.05

    cat(sprintf("[%s] %s null KS p = %.3f (> 0.05)\n",
                ifelse(ok_null, "OK", "WARN"), test, ks_p))
    cat(sprintf("[%s] %s null rejection rate = %.3f (<= 0.10)\n",
                ifelse(ok_null_rate, "OK", "WARN"), test, null_rej))
    cat(sprintf("[%s] %s collider rejection rate = %.3f (> 0.05)\n",
                ifelse(ok_alt, "OK", "WARN"), test, alt_rej))
  }
}

# Flag any extreme values
for (route in c("A", "B")) {
  route_summary <- ci_summary[ci_summary$route == route, ]
  for (i in seq_len(nrow(route_summary))) {
    if (route_summary$null_ks_p[i] < 1e-3) {
      cat(sprintf("[FLAG] Route %s %s: null KS p = %.6f (< 1e-3)\n",
                  route, route_summary$test[i], route_summary$null_ks_p[i]))
    }
  }
}

cat("\n=== END SELF-DIAGNOSIS ===\n")

# =============================================================================
# Generate LaTeX Table for Paper (Route A vs Route B CI Diagnostics)
# =============================================================================

cat("\nGenerating LaTeX table for Route A vs Route B CI diagnostics...\n")

tables_dir <- "../Hybrid-Frugal-Paper/tables"
if (!dir.exists(tables_dir)) dir.create(tables_dir, recursive = TRUE)

# Read the summary CSV we just wrote
ci_summary <- read.csv("results/static_routeA_vs_routeB_ci_summary.csv",
                        stringsAsFactors = FALSE)

# Test display names
test_labels <- c(pcor = "Partial cor.\\ ", gcm = "GCM", rcot = "RCoT")

tex_lines <- c(
  "\\begin{table}[htbp]",
  "\\centering",
  paste0("\\caption{Route A vs Route B: CI test summary for the static $\\mathcal{M}_A$ model. ",
         "Null KS $p$: Kolmogorov--Smirnov test for uniformity of null p-values (larger is better). ",
         "The null rejection rate represents fraction of replications rejecting ",
         "$Z_2 \\perp\\!\\!\\!\\perp Y \\mid Z_1, Z_3$ at $\\alpha = 0.05$ (target $\\approx 0.05$). ",
         "The collider rejection rate represents fraction rejecting ",
         "$Z_1 \\perp\\!\\!\\!\\perp Z_3 \\mid Y, Z_2$, with larger values indicating greater power. }"),
  "\\label{tab:routeAB-ci-summary}",
  "\\begin{tabular}{llccc}",
  "\\toprule",
  "\\textbf{Route} & \\textbf{Test} & \\textbf{Null KS $p$} & \\textbf{Null rej.\\ rate} & \\textbf{Collider rej.\\ rate} \\\\",
  "\\midrule"
)

for (route in c("A", "B")) {
  route_data <- ci_summary[ci_summary$route == route, ]
  for (i in seq_along(test_labels)) {
    test_name <- names(test_labels)[i]
    label <- test_labels[i]
    row <- route_data[route_data$test == test_name, ]
    tex_lines <- c(tex_lines, sprintf(
      "%s & %s & %.3f & %.3f & %.3f \\\\",
      route, label, row$null_ks_p, row$null_reject_rate, row$alt_reject_rate
    ))
  }
  if (route == "A") {
    tex_lines <- c(tex_lines, "\\addlinespace")
  }
}

tex_lines <- c(tex_lines,
  "\\bottomrule",
  "\\end{tabular}",
  "\\end{table}"
)

tex_path <- file.path(tables_dir, "static_routeAB_ci_summary.tex")
writeLines(tex_lines, tex_path)
cat(sprintf("  Saved %s\n", tex_path))
