#' @title Generate Frugal Outcome with Known Causal Margin
#' @description Functions for generating outcome variables with explicitly specified
#'   causal margins p(Y|do(X)) while encoding Y-Z dependence via Gaussian copula.
#' @name generate_frugal_outcome
NULL

#' Generate Frugal Outcome with Known Causal Margin
#'
#' Generates outcome Y with a specified causal margin p(Y|do(X)) and copula
#' dependence with confounders Z. This is the core function for frugal
#' parameterization where the causal effect is explicitly specified.
#'
#' The key insight is that:
#' - X affects Y ONLY through the causal margin (mean function)
#' - The copula encodes Y-Z dependence conditional on X
#' - This ensures p(Y|do(X)) = integral p(Y|X,Z) p(Z) dZ is correctly specified
#'
#' @param X Numeric vector. Treatment variable (typically binary 0/1).
#' @param Z Matrix or data.frame. Confounder variables.
#' @param causal_mean_fn Function. Takes X as input and returns E[Y|do(X=x)].
#'   For example: function(x) beta0 + beta1 * x
#' @param causal_sd Numeric. Standard deviation of Y|do(X). Default is 1.
#' @param rho_Y_Z Numeric vector. Partial correlations between Y and each column
#'   of Z (conditional on all previous Z columns in topological order).
#'   Length must equal ncol(Z).
#' @param outcome_family Character. Distribution family for Y. Currently only
#'   "gaussian" is supported. Default is "gaussian".
#' @param seed Optional integer for reproducibility.
#'
#' @return A list containing:
#' \describe{
#'   \item{Y}{The generated outcome vector}
#'   \item{Y_ranks}{The outcome ranks (uniform on [0,1])}
#'   \item{causal_means}{The causal means E[Y|do(X)] for each observation}
#'   \item{fitted_copula}{The fitted Gaussian copula to Z}
#'   \item{full_correlation_matrix}{The full correlation matrix including Y}
#' }
#'
#' @details
#' The function works by:
#' 1. Fitting a Gaussian copula to the observed confounders Z
#' 2. Computing marginal ranks of Z (unconditioning if necessary)
#' 3. Building the full correlation matrix including Y using the specified
#'    partial correlations rho_Y_Z
#' 4. Generating Y conditional on (marginal ranks of Z) with:
#'    - Copula dependence from the Gaussian copula structure
#'    - Marginal distribution determined by causal_mean_fn(X) and causal_sd
#'
#' IMPORTANT: X does NOT enter the copula structure. The treatment effect
#' operates solely through the causal margin specification.
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' N <- 1000
#'
#' # Generate confounders
#' Z <- rgamma(N, shape = 2, scale = 2)
#'
#' # Generate treatment with confounding
#' ps <- plogis(-0.5 + 0.4 * scale(Z))
#' X <- rbinom(N, 1, ps)
#'
#' # Specify causal margin: E[Y|do(X=x)] = -0.5 + 0.3*x
#' causal_fn <- function(x) -0.5 + 0.3 * x
#'
#' # Generate Y with known causal effect
#' result <- generate_frugal_outcome(
#'   X = X,
#'   Z = as.matrix(Z),
#'   causal_mean_fn = causal_fn,
#'   causal_sd = 1,
#'   rho_Y_Z = 0.5
#' )
#'
#' # True ATE = 0.3
#' }
#'
#' @importFrom stats qnorm pnorm rnorm
#' @export
generate_frugal_outcome <- function(X,
                                     Z,
                                     causal_mean_fn,
                                     causal_sd = 1,
                                     rho_Y_Z,
                                     outcome_family = "gaussian",
                                     seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  # Input validation
  Z <- as.matrix(Z)
  N <- length(X)
  n_confounders <- ncol(Z)

  if (nrow(Z) != N) {
    stop("Number of rows in Z must match length of X")
  }

  if (length(rho_Y_Z) != n_confounders) {
    stop("Length of rho_Y_Z must equal number of confounders (columns in Z)")
  }

  if (outcome_family != "gaussian") {
    stop("Currently only 'gaussian' outcome family is supported")
  }

  # Step 1: Convert Z to empirical quantiles (marginal ranks)
  Z_ranks <- apply(Z, 2, function(col) {
    # Use n/(n+1) adjustment to avoid 0 and 1
    rank(col, ties.method = "average") / (N + 1)
  })
  Z_ranks <- as.matrix(Z_ranks)  # Ensure it's a matrix even for single column

  # Step 2: Handle single vs multiple confounders
  if (n_confounders == 1) {
    # Single confounder case: no copula fitting needed between Z's
    # The correlation matrix is just [1, rho; rho, 1] for (Z, Y)
    fitted_copula <- list(correlationMatrix = matrix(1, 1, 1))

    # Full correlation matrix is 2x2: (Z, Y)
    fullCorrelationMatrix <- matrix(c(1, rho_Y_Z[1], rho_Y_Z[1], 1), nrow = 2)
  } else {
    # Multiple confounders: fit Gaussian copula to Z
    fitted_copula <- fitMVGaussianCopula(dataQuantiles = Z_ranks, method = 'itau')

    # Build full correlation matrix including Y
    # The topological order for the vine: Z1, Z2, ..., Zk, Y
    topoOrder <- 1:n_confounders
    fullCorrelationMatrix <- computeFullCorMatrix(
      topoOrder = topoOrder,
      corMatrixMN = fitted_copula$correlationMatrix,
      vineCorParams = rho_Y_Z
    )
  }

  # Step 3: Generate Y conditional on Z ranks using the copula structure
  # Transform Z ranks to normal scale
  Z_normal <- qnorm(Z_ranks)

  # Generate Y from conditional normal distribution
  # Y | Z ~ N(mu_Y|Z, sigma_Y|Z) where the copula encodes the correlation
  outcome_model <- multivariate_conditional_mean_and_samples(
    X2_samples = Z_normal,
    R = fullCorrelationMatrix
  )

  # outcome_model gives us Y on standard normal scale (mean 0, var = conditional var)
  # We need to transform this to have the specified causal margin

  # Step 5: Transform to have correct causal margin
  # The copula-generated Y is standard normal-ish conditional on Z
  # We want: Y | do(X=x) ~ N(causal_mean_fn(x), causal_sd^2)
  #
  # Key: The copula quantile U_Y encodes Y-Z dependence
  # We use: Y = F_Y^{-1}(U_Y; X) where F_Y depends on X through causal_mean_fn

  # Get the copula ranks for Y (these encode Y-Z dependence, NOT X dependence)
  Y_ranks <- as.vector(outcome_model$generated_samples)
  Y_ranks <- pnorm(Y_ranks)  # Convert back to uniform

  # Transform using the causal margin: Y = causal_mean + causal_sd * qnorm(U_Y)
  causal_means <- causal_mean_fn(X)
  Y <- causal_means + causal_sd * qnorm(Y_ranks)

  return(list(
    Y = Y,
    Y_ranks = Y_ranks,
    causal_means = causal_means,
    fitted_copula = fitted_copula,
    full_correlation_matrix = fullCorrelationMatrix
  ))
}


#' Generate Frugal Outcome from Conditional Ranks (Approach A)
#'
#' Generates outcome Y using "Approach A" from the nonparanormal approximation:
#' - Regenerates Z from a fitted Gaussian copula (ensures perfect Markov structure)
#' - Generates Y with specified causal margin
#'
#' This approach is preferred when exact Markov structure is required.
#'
#' @param X Numeric vector. Treatment variable (typically binary 0/1).
#' @param Z_original Matrix or data.frame. Original confounder data (used to fit copula).
#' @param Z_cond_ranks Matrix. Conditional ranks from the BN simulation
#'   (e.g., U1, U_{2|1}, U_{3|1,2}, ...).
#' @param dag_parents List. Parent structure for the DAG. Each element is a vector
#'   of parent indices for that variable.
#' @param causal_mean_fn Function. Takes X as input and returns E[Y|do(X=x)].
#' @param causal_sd Numeric. Standard deviation of Y|do(X). Default is 1.
#' @param rho_Y_Z Numeric vector. Partial correlations between Y and each Z.
#' @param outcome_family Character. Currently only "gaussian". Default is "gaussian".
#' @param seed Optional integer for reproducibility.
#'
#' @return A list containing:
#' \describe{
#'   \item{Y}{The generated outcome vector}
#'   \item{Y_ranks}{The outcome ranks}
#'   \item{Z_regenerated}{The regenerated Z values (with exact Markov structure)}
#'   \item{Z_marginal_ranks}{Marginal ranks of Z (after unconditioning)}
#'   \item{causal_means}{The causal means E[Y|do(X)] for each observation}
#'   \item{fitted_copula}{The fitted Gaussian copula}
#'   \item{full_correlation_matrix}{The full correlation matrix including Y}
#' }
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' N <- 1000
#'
#' # Generate from BN: Z1 -> Z2
#' U1 <- runif(N)
#' Z1 <- qgamma(U1, shape = 2, scale = 2)
#' U2_1 <- runif(N)  # Conditional rank
#' Z2 <- qgamma(U2_1, shape = 2 + 1.5 * Z1, scale = 1)
#'
#' Z_original <- cbind(Z1, Z2)
#' Z_cond_ranks <- cbind(U1, U2_1)
#'
#' # Treatment with confounding
#' ps <- plogis(-0.5 + 0.2 * scale(Z1) + 0.3 * scale(Z2))
#' X <- rbinom(N, 1, ps)
#'
#' # DAG structure
#' dag_parents <- list(integer(0), c(1))
#'
#' result <- generate_frugal_outcome_approach_A(
#'   X = X,
#'   Z_original = Z_original,
#'   Z_cond_ranks = Z_cond_ranks,
#'   dag_parents = dag_parents,
#'   causal_mean_fn = function(x) -0.5 + 0.3 * x,
#'   causal_sd = 1,
#'   rho_Y_Z = c(0.4, 0.3)
#' )
#' }
#'
#' @importFrom stats qnorm pnorm rnorm qgamma
#' @export
generate_frugal_outcome_approach_A <- function(X,
                                                Z_original,
                                                Z_cond_ranks,
                                                dag_parents,
                                                causal_mean_fn,
                                                causal_sd = 1,
                                                rho_Y_Z,
                                                outcome_family = "gaussian",
                                                seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  # Input validation
  Z_original <- as.matrix(Z_original)
  Z_cond_ranks <- as.matrix(Z_cond_ranks)
  N <- length(X)
  n_confounders <- ncol(Z_original)

  if (nrow(Z_original) != N || nrow(Z_cond_ranks) != N) {
    stop("Number of rows in Z_original and Z_cond_ranks must match length of X")
  }

  if (length(rho_Y_Z) != n_confounders) {
    stop("Length of rho_Y_Z must equal number of confounders")
  }

  if (length(dag_parents) != n_confounders) {
    stop("Length of dag_parents must equal number of confounders")
  }

  # Step 1: Convert Z_original to empirical quantiles for copula fitting
  Z_ranks <- apply(Z_original, 2, function(col) {
    rank(col, ties.method = "average") / (N + 1)
  })

  # Step 2: Fit Gaussian copula to Z_original
  fitted_copula <- fitMVGaussianCopula(dataQuantiles = Z_ranks, method = 'itau')

  # Step 3: Uncondition the conditional ranks to get marginal ranks
  Z_marginal_ranks <- uncondition_conditional_ranks(
    cond_ranks = Z_cond_ranks,
    R = fitted_copula$correlationMatrix,
    parents = dag_parents
  )

  # Step 4: Build full correlation matrix including Y
  topoOrder <- 1:n_confounders
  fullCorrelationMatrix <- computeFullCorMatrix(
    topoOrder = topoOrder,
    corMatrixMN = fitted_copula$correlationMatrix,
    vineCorParams = rho_Y_Z
  )

  # Step 5: Generate Y conditional on marginal Z ranks
  Z_normal <- qnorm(Z_marginal_ranks)

  outcome_model <- multivariate_conditional_mean_and_samples(
    X2_samples = Z_normal,
    R = fullCorrelationMatrix
  )

  # Step 6: Transform to have correct causal margin
  Y_ranks <- pnorm(as.vector(outcome_model$generated_samples))
  causal_means <- causal_mean_fn(X)
  Y <- causal_means + causal_sd * qnorm(Y_ranks)

  # Optional: Regenerate Z from the Gaussian copula (for perfect Markov structure)
  # This would use the marginal ranks with the original quantile functions
  # For now, we return the original Z

  return(list(
    Y = Y,
    Y_ranks = Y_ranks,
    Z_marginal_ranks = Z_marginal_ranks,
    causal_means = causal_means,
    fitted_copula = fitted_copula,
    full_correlation_matrix = fullCorrelationMatrix
  ))
}


#' Estimate Propensity Scores
#'
#' Fits a logistic regression model to estimate propensity scores P(X=1|Z).
#'
#' @param X Numeric vector. Binary treatment (0/1).
#' @param Z Matrix or data.frame. Confounders.
#' @param formula_rhs Optional character string specifying the right-hand side
#'   of the formula. If NULL (default), uses all columns of Z linearly.
#'
#' @return A list containing:
#' \describe{
#'   \item{ps}{Estimated propensity scores}
#'   \item{model}{The fitted glm object}
#' }
#'
#' @importFrom stats glm binomial fitted
#' @export
estimate_propensity_scores <- function(X, Z, formula_rhs = NULL) {
  Z <- as.data.frame(Z)

  if (is.null(formula_rhs)) {
    # Use all columns linearly
    colnames(Z) <- paste0("Z", 1:ncol(Z))
    formula_str <- paste("X ~", paste(colnames(Z), collapse = " + "))
  } else {
    formula_str <- paste("X ~", formula_rhs)
  }

  data <- cbind(X = X, Z)
  model <- glm(as.formula(formula_str), data = data, family = binomial())
  ps <- fitted(model)

  return(list(ps = ps, model = model))
}


#' Compute IPW Estimator for ATE
#'
#' Computes the Inverse Probability Weighted (IPW) estimator for the
#' Average Treatment Effect.
#'
#' @param Y Numeric vector. Outcome variable.
#' @param X Numeric vector. Binary treatment (0/1).
#' @param ps Numeric vector. Propensity scores P(X=1|Z).
#' @param stabilized Logical. If TRUE, uses stabilized weights. Default is TRUE.
#'
#' @return A list containing:
#' \describe{
#'   \item{ate}{The IPW estimate of ATE}
#'   \item{mu1}{Estimated E[Y|do(X=1)]}
#'   \item{mu0}{Estimated E[Y|do(X=0)]}
#'   \item{weights}{The IPW weights used}
#' }
#'
#' @export
compute_ipw_ate <- function(Y, X, ps, stabilized = TRUE) {
  # Clip propensity scores to avoid extreme weights
  ps <- pmax(pmin(ps, 0.99), 0.01)

  if (stabilized) {
    # Stabilized weights
    p_X1 <- mean(X)
    p_X0 <- 1 - p_X1
    w1 <- p_X1 / ps
    w0 <- p_X0 / (1 - ps)
  } else {
    w1 <- 1 / ps
    w0 <- 1 / (1 - ps)
  }

  # IPW estimates
  mu1 <- sum(X * w1 * Y) / sum(X * w1)
  mu0 <- sum((1 - X) * w0 * Y) / sum((1 - X) * w0)
  ate <- mu1 - mu0

  return(list(
    ate = ate,
    mu1 = mu1,
    mu0 = mu0,
    weights = list(w1 = w1, w0 = w0)
  ))
}


#' Compute G-computation Estimator for ATE
#'
#' Computes the G-computation (outcome regression) estimator for the ATE.
#'
#' @param Y Numeric vector. Outcome variable.
#' @param X Numeric vector. Binary treatment (0/1).
#' @param Z Matrix or data.frame. Confounders.
#' @param formula_rhs Optional character string for outcome model RHS.
#'   If NULL, uses X + all Z columns linearly.
#'
#' @return A list containing:
#' \describe{
#'   \item{ate}{The G-computation estimate of ATE}
#'   \item{mu1}{Estimated E[Y|do(X=1)]}
#'   \item{mu0}{Estimated E[Y|do(X=0)]}
#'   \item{model}{The fitted lm object}
#' }
#'
#' @importFrom stats lm predict
#' @export
compute_gcomp_ate <- function(Y, X, Z, formula_rhs = NULL) {
  Z <- as.data.frame(Z)
  colnames(Z) <- paste0("Z", 1:ncol(Z))

  if (is.null(formula_rhs)) {
    formula_str <- paste("Y ~ X +", paste(colnames(Z), collapse = " + "))
  } else {
    formula_str <- paste("Y ~", formula_rhs)
  }

  data <- cbind(Y = Y, X = X, Z)
  model <- lm(as.formula(formula_str), data = data)

  # Predict under X=1 and X=0 for all observations
  data_X1 <- data
  data_X1$X <- 1
  data_X0 <- data
  data_X0$X <- 0

  Y_hat_1 <- predict(model, newdata = data_X1)
  Y_hat_0 <- predict(model, newdata = data_X0)

  mu1 <- mean(Y_hat_1)
  mu0 <- mean(Y_hat_0)
  ate <- mu1 - mu0

  return(list(
    ate = ate,
    mu1 = mu1,
    mu0 = mu0,
    model = model
  ))
}


#' Compute Doubly Robust (AIPW) Estimator for ATE
#'
#' Computes the Augmented Inverse Probability Weighted (AIPW) estimator,
#' also known as the doubly robust estimator.
#'
#' @param Y Numeric vector. Outcome variable.
#' @param X Numeric vector. Binary treatment (0/1).
#' @param Z Matrix or data.frame. Confounders.
#' @param ps Numeric vector. Propensity scores P(X=1|Z).
#' @param outcome_formula_rhs Optional formula RHS for outcome model.
#'
#' @return A list containing:
#' \describe{
#'   \item{ate}{The AIPW estimate of ATE}
#'   \item{mu1}{Estimated E[Y|do(X=1)]}
#'   \item{mu0}{Estimated E[Y|do(X=0)]}
#'   \item{outcome_model}{The fitted outcome model}
#' }
#'
#' @importFrom stats lm predict
#' @export
compute_aipw_ate <- function(Y, X, Z, ps, outcome_formula_rhs = NULL) {
  Z <- as.data.frame(Z)
  colnames(Z) <- paste0("Z", 1:ncol(Z))

  # Clip propensity scores
  ps <- pmax(pmin(ps, 0.99), 0.01)

  # Fit outcome model
  if (is.null(outcome_formula_rhs)) {
    formula_str <- paste("Y ~ X +", paste(colnames(Z), collapse = " + "))
  } else {
    formula_str <- paste("Y ~", outcome_formula_rhs)
  }

  data <- cbind(Y = Y, X = X, Z)
  outcome_model <- lm(as.formula(formula_str), data = data)

  # Predict potential outcomes
  data_X1 <- data
  data_X1$X <- 1
  data_X0 <- data
  data_X0$X <- 0

  mu_hat_1 <- predict(outcome_model, newdata = data_X1)
  mu_hat_0 <- predict(outcome_model, newdata = data_X0)

  # AIPW estimator
  n <- length(Y)

  # E[Y(1)]
  phi_1 <- mu_hat_1 + X * (Y - mu_hat_1) / ps
  mu1 <- mean(phi_1)

  # E[Y(0)]
  phi_0 <- mu_hat_0 + (1 - X) * (Y - mu_hat_0) / (1 - ps)
  mu0 <- mean(phi_0)

  ate <- mu1 - mu0

  return(list(
    ate = ate,
    mu1 = mu1,
    mu0 = mu0,
    outcome_model = outcome_model
  ))
}


#' Compute Naive OLS Estimator (Biased under Confounding)
#'
#' Computes the naive OLS estimator that regresses Y on X only (ignoring confounders).
#' This estimator is expected to be biased when there is confounding.
#'
#' @param Y Numeric vector. Outcome variable.
#' @param X Numeric vector. Binary treatment (0/1).
#'
#' @return A list containing:
#' \describe{
#'   \item{ate}{The naive estimate (biased)}
#'   \item{model}{The fitted lm object}
#' }
#'
#' @importFrom stats lm coef
#' @export
compute_naive_ate <- function(Y, X) {
  model <- lm(Y ~ X)
  ate <- coef(model)["X"]

  return(list(
    ate = as.numeric(ate),
    model = model
  ))
}
