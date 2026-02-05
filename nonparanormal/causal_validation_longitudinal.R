#' =============================================================================
#' Causal Effect Validation Experiment - Longitudinal (Two Covariates per Time)
#' =============================================================================
#'
#' This script validates the nonparanormal approximation for longitudinal settings
#' with two covariates per time slice following a chain structure: Z1_t -> Z2_t.
#'
#' DGP Structure (generic time notation: t-1, t):
#'
#' Time t-1:                              Time t:
#' Z1_{t-1} --> Z2_{t-1}                 Z1_t --> Z2_t
#'     |           |                       |        |
#'     +-----+-----+---> X_{t-1}          +----+---+--> X_t
#'     |           |       |               |        |      |
#'     +----> Y_{t-1} <----+               +---> Y_t <-----+
#'                 |                                  ^
#'                 +----------------------------------+
#'
#' Key features:
#' - Two covariates per time: Z1_t (root) and Z2_t (depends on Z1_t)
#' - Treatment at time t (X_t is the intervention of interest)
#' - Y_{t-1} acts as a confounder for the Y_t ~ X_t relationship
#' - Tests Markov property: Y_t ⊥ Z^{t-1} | Z^t, X_t, Y_{t-1}
#' - Includes GCM tests for conditional independence validation
#'
#' =============================================================================

library(tidyverse)
library(ggplot2)
library(ppcor)

# Source the nonparanormal functions
source("nonparanormal.R")
source("R/generate_frugal_outcome.R")

# Source the new DAG-constrained Gaussian copula functions
source("R/uniform_transforms.R")
source("R/gaussian_copula_dag.R")
source("R/gaussian_copula_dag_sample.R")

# Try to load GCM package (may not be available)
# NOTE: Disabled due to segfaults in parallel execution
gcm_available <- FALSE
# gcm_available <- requireNamespace("GeneralisedCovarianceMeasure", quietly = TRUE)
# if (gcm_available) {
#   library(GeneralisedCovarianceMeasure)
# }

# =============================================================================
# Experiment Parameters
# =============================================================================

# Simulation settings
N_SAMPLES <- 5000        # Sample size per simulation
N_SIMS <- 200            # Number of Monte Carlo simulations
SEED_BASE <- 123         # Base seed for reproducibility

# -----------------------------------------------------------------------------
# Time t-1 Parameters (CONDITIONAL marginal distributions)
# -----------------------------------------------------------------------------
# Z1_{t-1} distribution (root covariate - unconditional)
Z1_TM1_SHAPE <- 2
Z1_TM1_SCALE <- 2

# Z2_{t-1} | Z1_{t-1} distribution (conditional on Z1_{t-1})
Z2_TM1_BASE_SHAPE <- 2
Z2_TM1_SCALE <- 2
Z2_TM1_Z1_EFFECT <- 0.5     # Shape = base + effect * Z1_{t-1}

# X_{t-1} propensity (not used for causal estimation, just generates data)
ALPHA_X_TM1_0 <- -0.3       # Intercept
ALPHA_X_TM1_Z1 <- 0.4       # Z1_{t-1} effect
ALPHA_X_TM1_Z2 <- 0.3       # Z2_{t-1} effect

# Y_{t-1} causal parameters (Y_{t-1} | do(X_{t-1}))
BETA_Y_TM1_0 <- 0.0         # Intercept
BETA_Y_TM1_X <- 0.4         # X_{t-1} effect on Y_{t-1}
Y_TM1_SD <- 1.0

# Y_{t-1}-(Z1_{t-1}, Z2_{t-1}) copula partial correlations (for vine extension)
RHO_Y_TM1_Z1 <- 0.5         # Y_{t-1}-Z1_{t-1} partial correlation
RHO_Y_TM1_Z2 <- 0.4         # Y_{t-1}-Z2_{t-1}|Z1_{t-1} partial correlation

# -----------------------------------------------------------------------------
# Time t Parameters (CONDITIONAL marginal distributions)
# -----------------------------------------------------------------------------
# Z1_t | Z1_{t-1} distribution (AR structure)
Z1_T_BASE_SHAPE <- 2
Z1_T_SCALE <- 1.5
Z1_T_AR_EFFECT <- 0.4       # Shape = base + effect * Z1_{t-1}

# Z2_t | Z2_{t-1}, Z1_t distribution (AR + chain structure)
Z2_T_BASE_SHAPE <- 2
Z2_T_SCALE <- 1.5
Z2_T_AR_EFFECT <- 0.3       # Effect of Z2_{t-1} on shape
Z2_T_Z1_EFFECT <- 0.4       # Effect of Z1_t on shape (chain)

# X_t propensity (the treatment of interest)
ALPHA_X_T_0 <- -0.5         # Intercept
ALPHA_X_T_Z1 <- 0.5         # Z1_t effect
ALPHA_X_T_Z2 <- 0.4         # Z2_t effect
ALPHA_X_T_Y <- 0.3          # Y_{t-1} effect (confounding)

# Y_t causal parameters: E[Y_t | do(X_t=x), Y_{t-1}=y] = gamma0 + gamma1*x + gamma2*y
# This is the dynamic causal margin we want to preserve
GAMMA_Y_T_0 <- 0.0          # Intercept
GAMMA_Y_T_X <- 0.5          # CAUSAL EFFECT of X_t on Y_t (this is the target)
GAMMA_Y_T_Y_TM1 <- 0.3      # Effect of Y_{t-1} on Y_t (not intervened)
Y_T_SD <- 1.0

# Y_t-(Z1_t, Z2_t, Y_{t-1}) copula partial correlations (for vine extension)
RHO_Y_T_Z1 <- 0.4           # Y_t-Z1_t partial correlation
RHO_Y_T_Z2 <- 0.3           # Y_t-Z2_t|Z1_t partial correlation
RHO_Y_T_Y_TM1 <- 0.3        # Y_t-Y_{t-1}|Z1_t,Z2_t partial correlation

# True ATE for X_t (marginalizing over Y_{t-1})
TRUE_ATE_X_T <- GAMMA_Y_T_X

# =============================================================================
# Helper Functions for Longitudinal Setting
# =============================================================================

#' Generate data from the longitudinal DGP (two time points, two covariates each)
#'
#' This function uses the BN factorization for Z variables (conditional marginals
#' with INDEPENDENT conditional ranks) and a Gaussian copula ONLY for the Y-Z
#' dependence.
#'
#' KEY INSIGHT: For Z-Z dependence with conditional marginals, we DON'T need
#' a copula! The factorization is:
#'   p(Z1_{t-1}, Z2_{t-1}, Z1_t, Z2_t) = p(Z1_{t-1}) * p(Z2_{t-1}|Z1_{t-1}) * ...
#'
#' The conditional ranks U_{Z_j | pa(Z_j)} are INDEPENDENT - we sample them iid Uniform.
#' The dependence comes entirely from the conditional CDFs (shape parameters).
#'
#' For Y-Z dependence, we use a Gaussian copula to specify the partial correlations.
#'
#' This ensures Y_t ⊥ Z_{t-1} | Z_t, X_t, Y_{t-1} BY CONSTRUCTION.
#'
#' @param n Sample size
#' @param seed Random seed
#'
#' @return List with all generated data
generate_longitudinal_data <- function(n, seed) {
  set.seed(seed)

  # ===========================================================================
  # BN FACTORIZATION FOR Z's + GAUSSIAN COPULA FOR Y's
  #
  # For Z variables:
  #   - Sample conditional ranks U_{Z_j | pa(Z_j)} ~ iid Uniform(0,1)
  #   - Transform: Z_j = F^{-1}_{Z_j | pa(Z_j)}(U_j | Z_{pa(j)})
  #   - Dependence comes from conditional CDFs, NOT from correlated ranks
  #
  # For Y variables:
  #   - Y_{t-1} depends on (Z1_{t-1}, Z2_{t-1}) via a copula
  #   - Y_t depends on (Z1_t, Z2_t, Y_{t-1}) via a copula
  #   - Copula specified by partial correlations
  #
  # This ensures Markov property by construction!
  # ===========================================================================

  # =========================================================================
  # STEP 1: Generate Z variables using BN factorization
  # =========================================================================
  # All conditional ranks are INDEPENDENT Uniform(0,1)

  # Z1_{t-1}: root node (unconditional)
  U_Z1_tm1 <- runif(n)
  Z1_tm1 <- qgamma(U_Z1_tm1, shape = Z1_TM1_SHAPE, scale = Z1_TM1_SCALE)

  # Z2_{t-1} | Z1_{t-1}: shape depends on Z1_{t-1}
  U_Z2_tm1 <- runif(n)  # INDEPENDENT of U_Z1_tm1
  Z2_tm1_shape <- Z2_TM1_BASE_SHAPE + Z2_TM1_Z1_EFFECT * pmax(Z1_tm1, 0)
  Z2_tm1 <- qgamma(U_Z2_tm1, shape = Z2_tm1_shape, scale = Z2_TM1_SCALE)

  # Z1_t | Z1_{t-1}: AR structure via conditional CDF
  U_Z1_t <- runif(n)  # INDEPENDENT
  Z1_t_shape <- Z1_T_BASE_SHAPE + Z1_T_AR_EFFECT * pmax(Z1_tm1, 0)
  Z1_t <- qgamma(U_Z1_t, shape = Z1_t_shape, scale = Z1_T_SCALE)

  # Z2_t | Z2_{t-1}, Z1_t: depends on both parents
  U_Z2_t <- runif(n)  # INDEPENDENT
  Z2_t_shape <- Z2_T_BASE_SHAPE + Z2_T_AR_EFFECT * pmax(Z2_tm1, 0) + Z2_T_Z1_EFFECT * pmax(Z1_t, 0)
  Z2_t <- qgamma(U_Z2_t, shape = Z2_t_shape, scale = Z2_T_SCALE)

  # =========================================================================
  # STEP 2: Generate Y_{t-1} using Gaussian copula with (Z1_{t-1}, Z2_{t-1})
  # =========================================================================
  # Y_{t-1} is correlated with Z_{t-1} through specified partial correlations.
  # We need to sample U_{Y_{t-1}} that has the right copula dependence with Z_{t-1}.

  # Compute marginal ranks of Z_{t-1}
  U_Z1_tm1_marg <- rank(Z1_tm1, ties.method = "average") / (n + 1)
  U_Z2_tm1_marg <- rank(Z2_tm1, ties.method = "average") / (n + 1)

  # Transform to Gaussian scale
  Q_Z1_tm1 <- qnorm(U_Z1_tm1_marg)
  Q_Z2_tm1 <- qnorm(U_Z2_tm1_marg)

  # Compute empirical correlation
  rho_Z1_Z2_tm1 <- cor(Q_Z1_tm1, Q_Z2_tm1)

  # Build correlation matrix for (Z1_{t-1}, Z2_{t-1}, Y_{t-1})
  # Partial correlations: rho(Y, Z1) = RHO_Y_TM1_Z1, rho(Y, Z2|Z1) = RHO_Y_TM1_Z2
  # Convert to marginal correlations using vine formula
  rho_Y_Z1_tm1 <- RHO_Y_TM1_Z1
  rho_Y_Z2_tm1 <- RHO_Y_TM1_Z2 * sqrt(1 - rho_Y_Z1_tm1^2) * sqrt(1 - rho_Z1_Z2_tm1^2) +
                  rho_Y_Z1_tm1 * rho_Z1_Z2_tm1

  # Sample Q_{Y_{t-1}} | Q_{Z1_{t-1}}, Q_{Z2_{t-1}} using Gaussian conditional
  # Q_Y | Q_Z ~ N(Sigma_{YZ} Sigma_{ZZ}^{-1} Q_Z, Sigma_{Y|Z})
  Q_Z_tm1 <- cbind(Q_Z1_tm1, Q_Z2_tm1)
  r_Y_Z_tm1 <- c(rho_Y_Z1_tm1, rho_Y_Z2_tm1)
  R_ZZ_tm1 <- matrix(c(1, rho_Z1_Z2_tm1, rho_Z1_Z2_tm1, 1), 2, 2)
  R_ZZ_tm1_inv <- solve(R_ZZ_tm1)

  cond_mean_Y_tm1 <- as.vector(Q_Z_tm1 %*% R_ZZ_tm1_inv %*% r_Y_Z_tm1)
  cond_var_Y_tm1 <- as.numeric(1 - t(r_Y_Z_tm1) %*% R_ZZ_tm1_inv %*% r_Y_Z_tm1)

  # Sample conditional Y_{t-1}
  Q_Y_tm1_innov <- rnorm(n)
  Q_Y_tm1 <- cond_mean_Y_tm1 + sqrt(cond_var_Y_tm1) * Q_Y_tm1_innov
  U_Y_tm1 <- pnorm(Q_Y_tm1)

  # =========================================================================
  # STEP 3: Generate X_{t-1} and transform Y_{t-1} through causal margin
  # =========================================================================
  Z1_tm1_std <- scale(Z1_tm1)
  Z2_tm1_std <- scale(Z2_tm1)
  ps_X_tm1 <- plogis(ALPHA_X_TM1_0 + ALPHA_X_TM1_Z1 * Z1_tm1_std + ALPHA_X_TM1_Z2 * Z2_tm1_std)
  X_tm1 <- rbinom(n, 1, ps_X_tm1)

  # Y_{t-1} = causal_margin + noise via copula
  causal_means_Y_tm1 <- BETA_Y_TM1_0 + BETA_Y_TM1_X * X_tm1
  Y_tm1 <- causal_means_Y_tm1 + Y_TM1_SD * qnorm(U_Y_tm1)

  # =========================================================================
  # STEP 4: Generate Y_t using Gaussian copula with (Z1_t, Z2_t, Y_{t-1}) ONLY
  # =========================================================================
  # CRITICAL: Y_t depends on (Z1_t, Z2_t, Y_{t-1}) only, NOT on Z_{t-1}.
  # This ensures the Markov property Y_t ⊥ Z_{t-1} | Z_t, Y_{t-1} by construction!

  # Compute marginal ranks of conditioning set
  U_Z1_t_marg <- rank(Z1_t, ties.method = "average") / (n + 1)
  U_Z2_t_marg <- rank(Z2_t, ties.method = "average") / (n + 1)
  U_Y_tm1_marg <- rank(Y_tm1, ties.method = "average") / (n + 1)

  # Transform to Gaussian scale
  Q_Z1_t <- qnorm(U_Z1_t_marg)
  Q_Z2_t <- qnorm(U_Z2_t_marg)
  Q_Y_tm1_t <- qnorm(U_Y_tm1_marg)

  # Compute empirical correlations among conditioning set
  rho_Z1_Z2_t <- cor(Q_Z1_t, Q_Z2_t)
  rho_Z1_Y_tm1 <- cor(Q_Z1_t, Q_Y_tm1_t)
  rho_Z2_Y_tm1 <- cor(Q_Z2_t, Q_Y_tm1_t)

  # Build 3x3 correlation matrix for conditioning set
  R_cond <- matrix(c(
    1, rho_Z1_Z2_t, rho_Z1_Y_tm1,
    rho_Z1_Z2_t, 1, rho_Z2_Y_tm1,
    rho_Z1_Y_tm1, rho_Z2_Y_tm1, 1
  ), 3, 3)

  # Convert partial correlations to marginal correlations for Y_t
  # Partial corrs: rho(Y_t, Z1_t) = RHO_Y_T_Z1
  #                rho(Y_t, Z2_t | Z1_t) = RHO_Y_T_Z2
  #                rho(Y_t, Y_{t-1} | Z1_t, Z2_t) = RHO_Y_T_Y_TM1

  # First partial = first marginal
  rho_Yt_Z1 <- RHO_Y_T_Z1

  # Second: marginal from partial
  rho_Yt_Z2 <- RHO_Y_T_Z2 * sqrt(1 - rho_Yt_Z1^2) * sqrt(1 - rho_Z1_Z2_t^2) +
               rho_Yt_Z1 * rho_Z1_Z2_t

  # Third: marginal from partial given two variables
  # Use the formula: rho(Y, W | V) => rho(Y, W) = partial * sqrt(var(Y|V)) * sqrt(var(W|V)) + cov(Y,V) * cov(W,V) / var(V)
  # For multivariate conditioning, use matrix formula
  R_Z <- matrix(c(1, rho_Z1_Z2_t, rho_Z1_Z2_t, 1), 2, 2)
  R_Z_inv <- solve(R_Z)
  r_Yt_Z <- c(rho_Yt_Z1, rho_Yt_Z2)
  r_Ytm1_Z <- c(rho_Z1_Y_tm1, rho_Z2_Y_tm1)

  var_Yt_given_Z <- as.numeric(1 - t(r_Yt_Z) %*% R_Z_inv %*% r_Yt_Z)
  var_Ytm1_given_Z <- as.numeric(1 - t(r_Ytm1_Z) %*% R_Z_inv %*% r_Ytm1_Z)

  # Partial to marginal: rho(Y_t, Y_{t-1}) = partial * sqrt(var_Yt|Z) * sqrt(var_Ytm1|Z) + regression_part
  cov_Yt_Ytm1_given_Z <- RHO_Y_T_Y_TM1 * sqrt(var_Yt_given_Z) * sqrt(var_Ytm1_given_Z)
  regression_part <- as.numeric(t(r_Yt_Z) %*% R_Z_inv %*% r_Ytm1_Z)
  rho_Yt_Ytm1 <- cov_Yt_Ytm1_given_Z + regression_part

  # Marginal correlations for Y_t
  r_Yt_cond <- c(rho_Yt_Z1, rho_Yt_Z2, rho_Yt_Ytm1)

  # Sample Q_{Y_t} | Q_{Z1_t}, Q_{Z2_t}, Q_{Y_{t-1}}
  Q_cond_set <- cbind(Q_Z1_t, Q_Z2_t, Q_Y_tm1_t)
  R_cond_inv <- solve(R_cond)

  cond_mean_Y_t <- as.vector(Q_cond_set %*% R_cond_inv %*% r_Yt_cond)
  cond_var_Y_t <- as.numeric(1 - t(r_Yt_cond) %*% R_cond_inv %*% r_Yt_cond)

  # Ensure valid variance
  if (cond_var_Y_t <= 0) {
    warning("Conditional variance for Y_t is non-positive, setting to small value")
    cond_var_Y_t <- 0.01
  }

  Q_Y_t_innov <- rnorm(n)
  Q_Y_t <- cond_mean_Y_t + sqrt(cond_var_Y_t) * Q_Y_t_innov
  U_Y_t <- pnorm(Q_Y_t)

  # =========================================================================
  # STEP 5: Generate X_t and transform Y_t through causal margin
  # =========================================================================
  Z1_t_std <- scale(Z1_t)
  Z2_t_std <- scale(Z2_t)
  Y_tm1_std <- scale(Y_tm1)
  ps_X_t_true <- plogis(ALPHA_X_T_0 + ALPHA_X_T_Z1 * Z1_t_std +
                        ALPHA_X_T_Z2 * Z2_t_std + ALPHA_X_T_Y * Y_tm1_std)
  X_t <- rbinom(n, 1, ps_X_t_true)

  # Y_t = causal_margin + noise via copula
  causal_means_Y_t <- GAMMA_Y_T_0 + GAMMA_Y_T_X * X_t + GAMMA_Y_T_Y_TM1 * Y_tm1
  Y_t <- causal_means_Y_t + Y_T_SD * qnorm(U_Y_t)

  # =========================================================================
  # Return all data
  # =========================================================================
  return(list(
    # Time t-1
    Z1_tm1 = Z1_tm1,
    Z2_tm1 = Z2_tm1,
    X_tm1 = X_tm1,
    Y_tm1 = Y_tm1,
    ps_X_tm1 = ps_X_tm1,
    # Time t
    Z1_t = Z1_t,
    Z2_t = Z2_t,
    X_t = X_t,
    Y_t = Y_t,
    ps_X_t_true = ps_X_t_true,
    # Ranks (for verification)
    Y_t_ranks = U_Y_t,
    # Copula info (for debugging)
    R_cond = R_cond,
    r_Yt_cond = r_Yt_cond,
    cond_var_Y_t = cond_var_Y_t
  ))
}


#' Estimate propensity scores for X_t given (Z1_t, Z2_t, Y_{t-1})
#'
#' @param X_t Treatment at time t
#' @param Z1_t Confounder 1 at time t
#' @param Z2_t Confounder 2 at time t
#' @param Y_tm1 Outcome at time t-1 (also a confounder)
#'
#' @return Estimated propensity scores
estimate_ps_X_t <- function(X_t, Z1_t, Z2_t, Y_tm1) {
  data <- data.frame(X_t = X_t, Z1_t = scale(Z1_t), Z2_t = scale(Z2_t), Y_tm1 = scale(Y_tm1))
  model <- glm(X_t ~ Z1_t + Z2_t + Y_tm1, data = data, family = binomial())
  return(fitted(model))
}


#' Compute IPW estimate for longitudinal ATE of X_t
#'
#' @param Y_t Outcome at time t
#' @param X_t Treatment at time t
#' @param ps Propensity scores P(X_t=1|Z_t, Y_{t-1})
#'
#' @return IPW estimate of E[Y_t|do(X_t=1)] - E[Y_t|do(X_t=0)]
compute_ipw_longitudinal <- function(Y_t, X_t, ps) {
  ps <- pmax(pmin(ps, 0.99), 0.01)

  # Stabilized weights
  p_X_t <- mean(X_t)
  w1 <- p_X_t / ps
  w0 <- (1 - p_X_t) / (1 - ps)

  mu1 <- sum(X_t * w1 * Y_t) / sum(X_t * w1)
  mu0 <- sum((1 - X_t) * w0 * Y_t) / sum((1 - X_t) * w0)

  return(list(ate = mu1 - mu0, mu1 = mu1, mu0 = mu0))
}


#' Compute G-computation estimate for longitudinal ATE
#'
#' @param Y_t Outcome at time t
#' @param X_t Treatment at time t
#' @param Z1_t Confounder 1 at time t
#' @param Z2_t Confounder 2 at time t
#' @param Y_tm1 Outcome/confounder from time t-1
#'
#' @return G-computation estimate
compute_gcomp_longitudinal <- function(Y_t, X_t, Z1_t, Z2_t, Y_tm1) {
  data <- data.frame(Y_t = Y_t, X_t = X_t, Z1_t = scale(Z1_t),
                     Z2_t = scale(Z2_t), Y_tm1 = scale(Y_tm1))
  model <- lm(Y_t ~ X_t + Z1_t + Z2_t + Y_tm1, data = data)

  data_X_t_1 <- data
  data_X_t_1$X_t <- 1
  data_X_t_0 <- data
  data_X_t_0$X_t <- 0

  Y_t_hat_1 <- predict(model, newdata = data_X_t_1)
  Y_t_hat_0 <- predict(model, newdata = data_X_t_0)

  mu1 <- mean(Y_t_hat_1)
  mu0 <- mean(Y_t_hat_0)

  return(list(ate = mu1 - mu0, mu1 = mu1, mu0 = mu0, model = model))
}


#' Compute AIPW estimate for longitudinal setting
#'
#' @param Y_t Outcome at time t
#' @param X_t Treatment at time t
#' @param Z1_t Confounder 1 at time t
#' @param Z2_t Confounder 2 at time t
#' @param Y_tm1 Outcome from time t-1
#' @param ps Propensity scores
#'
#' @return AIPW estimate
compute_aipw_longitudinal <- function(Y_t, X_t, Z1_t, Z2_t, Y_tm1, ps) {
  ps <- pmax(pmin(ps, 0.99), 0.01)

  data <- data.frame(Y_t = Y_t, X_t = X_t, Z1_t = scale(Z1_t),
                     Z2_t = scale(Z2_t), Y_tm1 = scale(Y_tm1))
  outcome_model <- lm(Y_t ~ X_t + Z1_t + Z2_t + Y_tm1, data = data)

  data_X_t_1 <- data
  data_X_t_1$X_t <- 1
  data_X_t_0 <- data
  data_X_t_0$X_t <- 0

  mu_hat_1 <- predict(outcome_model, newdata = data_X_t_1)
  mu_hat_0 <- predict(outcome_model, newdata = data_X_t_0)

  # AIPW
  phi_1 <- mu_hat_1 + X_t * (Y_t - mu_hat_1) / ps
  phi_0 <- mu_hat_0 + (1 - X_t) * (Y_t - mu_hat_0) / (1 - ps)

  mu1 <- mean(phi_1)
  mu0 <- mean(phi_0)

  return(list(ate = mu1 - mu0, mu1 = mu1, mu0 = mu0))
}


# =============================================================================
# Test Markov Property: Y_t ⊥ Z^{t-1} | Z^t, X_t, Y_{t-1}
# =============================================================================

#' Test the Markov property for the longitudinal model
#'
#' Under the correct specification, Y_t should be conditionally independent
#' of (Z1_{t-1}, Z2_{t-1}) given (Z1_t, Z2_t, X_t, Y_{t-1}).
#'
#' @param data Data list from generate_longitudinal_data
#'
#' @return List with test results
test_markov_property <- function(data) {
  n <- length(data$Y_t)

  # Test 1: Y_t ⊥ Z1_{t-1} | Z1_t, Z2_t, X_t, Y_{t-1}
  # Using partial correlation
  pcor_Z1_tm1 <- ppcor::pcor.test(
    data$Y_t, data$Z1_tm1,
    cbind(data$Z1_t, data$Z2_t, data$X_t, data$Y_tm1)
  )

  # Test 2: Y_t ⊥ Z2_{t-1} | Z1_t, Z2_t, X_t, Y_{t-1}
  pcor_Z2_tm1 <- ppcor::pcor.test(
    data$Y_t, data$Z2_tm1,
    cbind(data$Z1_t, data$Z2_t, data$X_t, data$Y_tm1)
  )

  return(list(
    # Z1_{t-1} test
    pcor_Z1_tm1_estimate = pcor_Z1_tm1$estimate,
    pcor_Z1_tm1_statistic = pcor_Z1_tm1$statistic,
    pcor_Z1_tm1_pvalue = pcor_Z1_tm1$p.value,
    # Z2_{t-1} test
    pcor_Z2_tm1_estimate = pcor_Z2_tm1$estimate,
    pcor_Z2_tm1_statistic = pcor_Z2_tm1$statistic,
    pcor_Z2_tm1_pvalue = pcor_Z2_tm1$p.value,
    # Summary
    markov_holds_Z1 = pcor_Z1_tm1$p.value > 0.05,
    markov_holds_Z2 = pcor_Z2_tm1$p.value > 0.05
  ))
}


#' Test Markov property using GCM (if available)
#'
#' @param data Data list from generate_longitudinal_data
#'
#' @return List with GCM test results
test_markov_gcm <- function(data) {
  if (!gcm_available) {
    return(list(
      gcm_Z1_pvalue = NA,
      gcm_Z2_pvalue = NA
    ))
  }

  # Conditioning set
  cond_set <- cbind(data$Z1_t, data$Z2_t, data$X_t, data$Y_tm1)

  # GCM test for Y_t ⊥ Z1_{t-1} | Z^t, X_t, Y_{t-1}
  gcm_Z1 <- tryCatch({
    GeneralisedCovarianceMeasure::gcm.test(
      X = data$Z1_tm1,
      Y = data$Y_t,
      Z = cond_set
    )
  }, error = function(e) list(p.value = NA))

  # GCM test for Y_t ⊥ Z2_{t-1} | Z^t, X_t, Y_{t-1}
  gcm_Z2 <- tryCatch({
    GeneralisedCovarianceMeasure::gcm.test(
      X = data$Z2_tm1,
      Y = data$Y_t,
      Z = cond_set
    )
  }, error = function(e) list(p.value = NA))

  return(list(
    gcm_Z1_pvalue = gcm_Z1$p.value,
    gcm_Z2_pvalue = gcm_Z2$p.value
  ))
}


#' Verify chain structure: Z1_t → Z2_t
#'
#' @param data Data list
#'
#' @return List with chain structure test results
verify_chain_structure <- function(data) {
  # Time t-1: Z2_{t-1} should depend on Z1_{t-1}
  cor_tm1 <- cor(data$Z1_tm1, data$Z2_tm1)

  # Time t: Z2_t should depend on Z1_t (partial corr given Z2_{t-1})
  pcor_t <- ppcor::pcor.test(data$Z2_t, data$Z1_t, data$Z2_tm1)

  return(list(
    cor_Z1_Z2_tm1 = cor_tm1,
    pcor_Z2t_Z1t_estimate = pcor_t$estimate,
    pcor_Z2t_Z1t_pvalue = pcor_t$p.value,
    chain_structure_exists = pcor_t$p.value < 0.05
  ))
}


#' Verify rank uniformity for Y_t
#'
#' @param data Data list
#'
#' @return KS test results
verify_rank_uniformity <- function(data) {
  ks_result <- ks.test(data$Y_t_ranks, "punif")
  return(list(
    ks_statistic = ks_result$statistic,
    ks_pvalue = ks_result$p.value,
    ranks_uniform = ks_result$p.value > 0.05
  ))
}


#' Run a single longitudinal simulation
#'
#' @param sim_id Simulation ID
#' @param n Sample size
#' @param verbose Print progress
#'
#' @return Data frame with results
run_single_longitudinal_simulation <- function(sim_id, n = N_SAMPLES, verbose = FALSE) {
  # Generate data
  data <- generate_longitudinal_data(n, seed = SEED_BASE + sim_id)

  # Estimate propensity scores for X_t
  ps_X_t_hat <- estimate_ps_X_t(data$X_t, data$Z1_t, data$Z2_t, data$Y_tm1)

  # Naive estimate (ignoring confounding)
  naive_model <- lm(data$Y_t ~ data$X_t)
  naive_ate <- coef(naive_model)["data$X_t"]

  # IPW estimate
  ipw_result <- compute_ipw_longitudinal(data$Y_t, data$X_t, ps_X_t_hat)

  # G-computation estimate
  gcomp_result <- compute_gcomp_longitudinal(data$Y_t, data$X_t, data$Z1_t,
                                              data$Z2_t, data$Y_tm1)

  # AIPW estimate
  aipw_result <- compute_aipw_longitudinal(data$Y_t, data$X_t, data$Z1_t,
                                            data$Z2_t, data$Y_tm1, ps_X_t_hat)

  # Markov property tests
  markov_result <- test_markov_property(data)
  gcm_result <- test_markov_gcm(data)

  # Chain structure verification
  chain_result <- verify_chain_structure(data)

  # Rank uniformity
  rank_result <- verify_rank_uniformity(data)

  results <- data.frame(
    sim_id = sim_id,
    n = n,
    true_ate = TRUE_ATE_X_T,
    naive_ate = as.numeric(naive_ate),
    ipw_ate = ipw_result$ate,
    gcomp_ate = gcomp_result$ate,
    aipw_ate = aipw_result$ate,
    # Markov tests (partial correlations)
    pcor_Z1_tm1 = markov_result$pcor_Z1_tm1_estimate,
    pcor_Z1_tm1_z = markov_result$pcor_Z1_tm1_statistic,
    pcor_Z1_tm1_p = markov_result$pcor_Z1_tm1_pvalue,
    pcor_Z2_tm1 = markov_result$pcor_Z2_tm1_estimate,
    pcor_Z2_tm1_z = markov_result$pcor_Z2_tm1_statistic,
    pcor_Z2_tm1_p = markov_result$pcor_Z2_tm1_pvalue,
    # GCM tests
    gcm_Z1_p = gcm_result$gcm_Z1_pvalue,
    gcm_Z2_p = gcm_result$gcm_Z2_pvalue,
    # Chain structure
    chain_cor = chain_result$cor_Z1_Z2_tm1,
    chain_pcor = chain_result$pcor_Z2t_Z1t_estimate,
    chain_pcor_p = chain_result$pcor_Z2t_Z1t_pvalue,
    # Rank uniformity
    rank_ks_p = rank_result$ks_pvalue,
    # Additional info
    prop_treated = mean(data$X_t),
    mean_ps = mean(ps_X_t_hat),
    stringsAsFactors = FALSE
  )

  if (verbose) {
    cat(sprintf("Sim %d: True=%.3f, Naive=%.3f, IPW=%.3f, GComp=%.3f, AIPW=%.3f | ",
                sim_id, TRUE_ATE_X_T, naive_ate, ipw_result$ate,
                gcomp_result$ate, aipw_result$ate))
    cat(sprintf("Markov: Z1_p=%.3f, Z2_p=%.3f\n",
                markov_result$pcor_Z1_tm1_pvalue, markov_result$pcor_Z2_tm1_pvalue))
  }

  return(results)
}


# =============================================================================
# Run Simulation Study
# =============================================================================

cat("=============================================================================\n")
cat("Causal Effect Validation Experiment - Longitudinal Model (Two Covariates)\n")
cat("=============================================================================\n")
cat(sprintf("Sample size: %d\n", N_SAMPLES))
cat(sprintf("Number of simulations: %d\n", N_SIMS))
cat(sprintf("True ATE (X_t on Y_t): %.3f\n", TRUE_ATE_X_T))
cat(sprintf("Dynamic causal margin: E[Y_t|do(X_t), Y_{t-1}] = %.1f + %.1f*X_t + %.1f*Y_{t-1}\n",
            GAMMA_Y_T_0, GAMMA_Y_T_X, GAMMA_Y_T_Y_TM1))
cat(sprintf("GCM package available: %s\n", gcm_available))
cat("=============================================================================\n\n")

# First, test structure verification on a single dataset
cat("Testing DGP structure on a single large dataset (N=10000)...\n")
test_data <- generate_longitudinal_data(10000, seed = 888)

# Chain structure
chain_test <- verify_chain_structure(test_data)
cat(sprintf("  Chain structure (Z1 -> Z2):\n"))
cat(sprintf("    Time t-1: cor(Z1, Z2) = %.3f\n", chain_test$cor_Z1_Z2_tm1))
cat(sprintf("    Time t: pcor(Z2_t, Z1_t | Z2_{t-1}) = %.3f (p = %.3f)\n",
            chain_test$pcor_Z2t_Z1t_estimate, chain_test$pcor_Z2t_Z1t_pvalue))

# Markov property
markov_test <- test_markov_property(test_data)
cat(sprintf("  Markov property (Y_t ⊥ Z^{t-1} | Z^t, X_t, Y_{t-1}):\n"))
cat(sprintf("    pcor(Y_t, Z1_{t-1} | cond.) = %.4f (z = %.2f, p = %.3f)\n",
            markov_test$pcor_Z1_tm1_estimate, markov_test$pcor_Z1_tm1_statistic,
            markov_test$pcor_Z1_tm1_pvalue))
cat(sprintf("    pcor(Y_t, Z2_{t-1} | cond.) = %.4f (z = %.2f, p = %.3f)\n",
            markov_test$pcor_Z2_tm1_estimate, markov_test$pcor_Z2_tm1_statistic,
            markov_test$pcor_Z2_tm1_pvalue))

# Rank uniformity
rank_test <- verify_rank_uniformity(test_data)
cat(sprintf("  Rank uniformity: KS test p = %.3f\n", rank_test$ks_pvalue))

cat("\n")

# Run all simulations
cat("Running simulations...\n")
pb <- txtProgressBar(min = 0, max = N_SIMS, style = 3)

results_list <- vector("list", N_SIMS)
for (i in 1:N_SIMS) {
  results_list[[i]] <- run_single_longitudinal_simulation(i, verbose = FALSE)
  setTxtProgressBar(pb, i)
}
close(pb)

# Combine results
results_df <- do.call(rbind, results_list)

# =============================================================================
# Summarize Results
# =============================================================================

cat("\n=============================================================================\n")
cat("RESULTS SUMMARY - LONGITUDINAL MODEL\n")
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

cat(sprintf("True ATE (X_t effect): %.4f\n\n", TRUE_ATE_X_T))

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
    mean_pcor_Z1 = mean(pcor_Z1_tm1),
    sd_pcor_Z1 = sd(pcor_Z1_tm1),
    mean_z_Z1 = mean(pcor_Z1_tm1_z),
    mean_p_Z1 = mean(pcor_Z1_tm1_p),
    sd_p_Z1 = sd(pcor_Z1_tm1_p),
    mean_pcor_Z2 = mean(pcor_Z2_tm1),
    sd_pcor_Z2 = sd(pcor_Z2_tm1),
    mean_z_Z2 = mean(pcor_Z2_tm1_z),
    mean_p_Z2 = mean(pcor_Z2_tm1_p),
    sd_p_Z2 = sd(pcor_Z2_tm1_p)
  )

cat(sprintf("Y_t ⊥ Z1_{t-1} | Z^t, X_t, Y_{t-1}:\n"))
cat(sprintf("  Mean partial corr: %.4f (SD: %.4f)\n",
            markov_summary$mean_pcor_Z1, markov_summary$sd_pcor_Z1))
cat(sprintf("  Mean z-statistic: %.2f\n", markov_summary$mean_z_Z1))
cat(sprintf("  Mean p-value: %.3f (SD: %.3f)\n",
            markov_summary$mean_p_Z1, markov_summary$sd_p_Z1))

cat(sprintf("\nY_t ⊥ Z2_{t-1} | Z^t, X_t, Y_{t-1}:\n"))
cat(sprintf("  Mean partial corr: %.4f (SD: %.4f)\n",
            markov_summary$mean_pcor_Z2, markov_summary$sd_pcor_Z2))
cat(sprintf("  Mean z-statistic: %.2f\n", markov_summary$mean_z_Z2))
cat(sprintf("  Mean p-value: %.3f (SD: %.3f)\n",
            markov_summary$mean_p_Z2, markov_summary$sd_p_Z2))

# KS test for uniformity of p-values
ks_Z1 <- ks.test(results_df$pcor_Z1_tm1_p, "punif")
ks_Z2 <- ks.test(results_df$pcor_Z2_tm1_p, "punif")

cat(sprintf("\nKS test for uniformity of p-values:\n"))
cat(sprintf("  Z1_{t-1}: p = %.3f\n", ks_Z1$p.value))
cat(sprintf("  Z2_{t-1}: p = %.3f\n", ks_Z2$p.value))

# GCM summary if available
if (gcm_available && sum(!is.na(results_df$gcm_Z1_p)) > 0) {
  gcm_summary <- results_df %>%
    filter(!is.na(gcm_Z1_p)) %>%
    summarise(
      mean_gcm_Z1 = mean(gcm_Z1_p, na.rm = TRUE),
      sd_gcm_Z1 = sd(gcm_Z1_p, na.rm = TRUE),
      mean_gcm_Z2 = mean(gcm_Z2_p, na.rm = TRUE),
      sd_gcm_Z2 = sd(gcm_Z2_p, na.rm = TRUE)
    )

  cat(sprintf("\nGCM Test p-values:\n"))
  cat(sprintf("  Y_t ⊥ Z1_{t-1}: Mean p = %.3f (SD: %.3f)\n",
              gcm_summary$mean_gcm_Z1, gcm_summary$sd_gcm_Z1))
  cat(sprintf("  Y_t ⊥ Z2_{t-1}: Mean p = %.3f (SD: %.3f)\n",
              gcm_summary$mean_gcm_Z2, gcm_summary$sd_gcm_Z2))

  ks_gcm_Z1 <- ks.test(results_df$gcm_Z1_p[!is.na(results_df$gcm_Z1_p)], "punif")
  ks_gcm_Z2 <- ks.test(results_df$gcm_Z2_p[!is.na(results_df$gcm_Z2_p)], "punif")

  cat(sprintf("  KS test for uniformity: Z1 p = %.3f, Z2 p = %.3f\n",
              ks_gcm_Z1$p.value, ks_gcm_Z2$p.value))
}

# =============================================================================
# Create Visualizations
# =============================================================================

# ATE boxplot
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

p_boxplot <- ggplot(plot_data, aes(x = estimator, y = estimate, fill = estimator)) +
  geom_boxplot(alpha = 0.7) +
  geom_hline(yintercept = TRUE_ATE_X_T, linetype = "dashed", color = "red", linewidth = 1) +
  annotate("text", x = 0.5, y = TRUE_ATE_X_T + 0.05,
           label = sprintf("True ATE = %.2f", TRUE_ATE_X_T),
           hjust = 0, color = "red", size = 4) +
  labs(
    x = "Estimator",
    y = "Estimated ATE (X_t on Y_t)",
    title = "Longitudinal Model: Causal Effect Estimates",
    subtitle = sprintf("N = %d, %d sims | Two covariates per time (Z1_t -> Z2_t chain)",
                       N_SAMPLES, N_SIMS)
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold", size = 14),
    axis.title = element_text(face = "bold")
  ) +
  scale_fill_brewer(palette = "Set2")

# P-value histograms for Markov tests
p_markov_Z1 <- ggplot(results_df, aes(x = pcor_Z1_tm1_p)) +
  geom_histogram(bins = 20, fill = "steelblue", color = "white", alpha = 0.7) +
  geom_hline(yintercept = N_SIMS / 20, linetype = "dashed", color = "red") +
  labs(
    x = "p-value",
    y = "Frequency",
    title = expression(paste("Markov test: ", Y[t], " ⊥ ", Z[paste("1,t-1")], " | ", Z^t, ", ", X[t], ", ", Y[t-1])),
    subtitle = sprintf("KS test for uniformity: p = %.3f", ks_Z1$p.value)
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 12))

p_markov_Z2 <- ggplot(results_df, aes(x = pcor_Z2_tm1_p)) +
  geom_histogram(bins = 20, fill = "steelblue", color = "white", alpha = 0.7) +
  geom_hline(yintercept = N_SIMS / 20, linetype = "dashed", color = "red") +
  labs(
    x = "p-value",
    y = "Frequency",
    title = expression(paste("Markov test: ", Y[t], " ⊥ ", Z[paste("2,t-1")], " | ", Z^t, ", ", X[t], ", ", Y[t-1])),
    subtitle = sprintf("KS test for uniformity: p = %.3f", ks_Z2$p.value)
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 12))

# GCM p-value histograms if available
if (gcm_available && sum(!is.na(results_df$gcm_Z1_p)) > 0) {
  p_gcm_Z1 <- ggplot(results_df %>% filter(!is.na(gcm_Z1_p)), aes(x = gcm_Z1_p)) +
    geom_histogram(bins = 20, fill = "darkgreen", color = "white", alpha = 0.7) +
    geom_hline(yintercept = sum(!is.na(results_df$gcm_Z1_p)) / 20,
               linetype = "dashed", color = "red") +
    labs(
      x = "p-value",
      y = "Frequency",
      title = expression(paste("GCM test: ", Y[t], " ⊥ ", Z[paste("1,t-1")], " | ", Z^t, ", ", X[t], ", ", Y[t-1])),
      subtitle = sprintf("KS test for uniformity: p = %.3f", ks_gcm_Z1$p.value)
    ) +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold", size = 12))

  p_gcm_Z2 <- ggplot(results_df %>% filter(!is.na(gcm_Z2_p)), aes(x = gcm_Z2_p)) +
    geom_histogram(bins = 20, fill = "darkgreen", color = "white", alpha = 0.7) +
    geom_hline(yintercept = sum(!is.na(results_df$gcm_Z2_p)) / 20,
               linetype = "dashed", color = "red") +
    labs(
      x = "p-value",
      y = "Frequency",
      title = expression(paste("GCM test: ", Y[t], " ⊥ ", Z[paste("2,t-1")], " | ", Z^t, ", ", X[t], ", ", Y[t-1])),
      subtitle = sprintf("KS test for uniformity: p = %.3f", ks_gcm_Z2$p.value)
    ) +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold", size = 12))
}

# =============================================================================
# Save Results
# =============================================================================

if (!dir.exists("results")) {
  dir.create("results")
}

ggsave("results/causal_validation_longitudinal_boxplot.png", p_boxplot,
       width = 10, height = 6, dpi = 300)
ggsave("results/gcm_pvalue_histogram_Z1.png", p_markov_Z1,
       width = 8, height = 5, dpi = 300)
ggsave("results/gcm_pvalue_histogram_Z2.png", p_markov_Z2,
       width = 8, height = 5, dpi = 300)

write.csv(results_df, "results/causal_validation_longitudinal_results.csv", row.names = FALSE)

summary_output <- list(
  parameters = list(
    n_samples = N_SAMPLES,
    n_sims = N_SIMS,
    true_ate = TRUE_ATE_X_T,
    gamma_0 = GAMMA_Y_T_0,
    gamma_x = GAMMA_Y_T_X,
    gamma_y_tm1 = GAMMA_Y_T_Y_TM1,
    rho_Y_Z1 = RHO_Y_T_Z1,
    rho_Y_Z2 = RHO_Y_T_Z2,
    rho_Y_Y_tm1 = RHO_Y_T_Y_TM1
  ),
  summary_stats = summary_stats,
  markov_summary = markov_summary,
  ks_tests = list(
    Z1_p = ks_Z1$p.value,
    Z2_p = ks_Z2$p.value
  ),
  hypothesis_tests = list(
    ipw = list(t = ipw_ttest$statistic, p = ipw_ttest$p.value),
    gcomp = list(t = gcomp_ttest$statistic, p = gcomp_ttest$p.value),
    aipw = list(t = aipw_ttest$statistic, p = aipw_ttest$p.value)
  )
)
saveRDS(summary_output, "results/causal_validation_longitudinal_summary.rds")

cat("\n=============================================================================\n")
cat("Results saved to ./results/\n")
cat("  - causal_validation_longitudinal_boxplot.png\n")
cat("  - gcm_pvalue_histogram_Z1.png\n")
cat("  - gcm_pvalue_histogram_Z2.png\n")
cat("  - causal_validation_longitudinal_results.csv\n")
cat("  - causal_validation_longitudinal_summary.rds\n")
cat("=============================================================================\n")

# =============================================================================
# Verification Checklist
# =============================================================================

cat("\n")
cat("=============================================================================\n")
cat("VERIFICATION CHECKLIST - LONGITUDINAL MODEL\n")
cat("=============================================================================\n")

# Check 1: Chain structure exists
chain_exists <- mean(results_df$chain_pcor_p < 0.05) > 0.9
cat(sprintf("[%s] Chain structure Z1_t -> Z2_t exists (%.0f%% significant)\n",
            ifelse(chain_exists, "PASS", "FAIL"),
            mean(results_df$chain_pcor_p < 0.05) * 100))

# Check 2: Markov property holds (p-values uniform)
markov_Z1_ok <- ks_Z1$p.value > 0.05
markov_Z2_ok <- ks_Z2$p.value > 0.05
cat(sprintf("[%s] Markov Y_t ⊥ Z1_{t-1} | cond (KS p = %.3f)\n",
            ifelse(markov_Z1_ok, "PASS", "WARN"), ks_Z1$p.value))
cat(sprintf("[%s] Markov Y_t ⊥ Z2_{t-1} | cond (KS p = %.3f)\n",
            ifelse(markov_Z2_ok, "PASS", "WARN"), ks_Z2$p.value))

# Check 3: Naive OLS biased
naive_biased <- abs(summary_stats$naive_bias) > 0.05
cat(sprintf("[%s] Naive OLS is biased (|bias| = %.3f > 0.05)\n",
            ifelse(naive_biased, "PASS", "FAIL"), abs(summary_stats$naive_bias)))

# Check 4-6: Estimators approximately unbiased
ipw_unbiased <- ipw_ttest$p.value > 0.05
gcomp_unbiased <- gcomp_ttest$p.value > 0.05
aipw_unbiased <- aipw_ttest$p.value > 0.05

cat(sprintf("[%s] IPW is unbiased (p = %.3f > 0.05)\n",
            ifelse(ipw_unbiased, "PASS", "WARN"), ipw_ttest$p.value))
cat(sprintf("[%s] G-computation is unbiased (p = %.3f > 0.05)\n",
            ifelse(gcomp_unbiased, "PASS", "WARN"), gcomp_ttest$p.value))
cat(sprintf("[%s] AIPW is unbiased (p = %.3f > 0.05)\n",
            ifelse(aipw_unbiased, "PASS", "WARN"), aipw_ttest$p.value))

# Check 7: Small bias magnitude
ipw_small_bias <- abs(summary_stats$ipw_bias) < 0.05
aipw_small_bias <- abs(summary_stats$aipw_bias) < 0.05

cat(sprintf("[%s] IPW has small bias (|bias| = %.3f < 0.05)\n",
            ifelse(ipw_small_bias, "PASS", "WARN"), abs(summary_stats$ipw_bias)))
cat(sprintf("[%s] AIPW has small bias (|bias| = %.3f < 0.05)\n",
            ifelse(aipw_small_bias, "PASS", "WARN"), abs(summary_stats$aipw_bias)))

# Check 8: Rank uniformity
rank_uniform <- mean(results_df$rank_ks_p > 0.05) > 0.9
cat(sprintf("[%s] Rank uniformity holds (%.0f%% pass KS test)\n",
            ifelse(rank_uniform, "PASS", "WARN"), mean(results_df$rank_ks_p > 0.05) * 100))

cat("=============================================================================\n")

# Print plots
print(p_boxplot)
print(p_markov_Z1)
print(p_markov_Z2)
