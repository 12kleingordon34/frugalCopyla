#' @title Outcome Generation Functions
#' @description Functions for generating outcome samples conditioned on covariates
#'   using the nonparanormal approximation.
#' @name outcome_generation
NULL

#' Generate Conditional Samples from Multivariate Normal
#'
#' Computes the conditional mean and variance of X1 given X2 from a multivariate
#' normal distribution and generates samples.
#'
#' @param X2_samples A matrix where rows are observations and columns are the
#'   conditioning variables (on the Gaussian scale).
#' @param R A correlation matrix for the full joint distribution.
#'
#' @return A list containing:
#' \describe{
#'   \item{generated_samples}{Samples from the conditional distribution}
#'   \item{conditional_means}{The conditional mean for each observation}
#'   \item{conditional_variance}{The conditional variance}
#'   \item{coeffs}{The regression coefficients}
#' }
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' X2 <- matrix(rnorm(200), ncol = 2)
#' R <- matrix(c(1, 0.5, 0.3, 0.5, 1, 0.4, 0.3, 0.4, 1), ncol = 3)
#' result <- multivariate_conditional_mean_and_samples(X2, R)
#' }
#'
#' @importFrom stats rnorm
#' @export
multivariate_conditional_mean_and_samples <- function(X2_samples, R) {
  k <- ncol(X2_samples)
  n_samples <- nrow(X2_samples)
  d <- nrow(R)

  X2_samples <- as.matrix(X2_samples)

  # Indices for partitioning the covariance matrix
  indices_1 <- d              # First variable (X1)
  indices_2 <- 1:(d - 1)      # Remaining variables (X2)

  # Extract the relevant components from the covariance matrix R
  R11 <- R[indices_1, indices_1]
  R12 <- matrix(R[indices_1, indices_2], nrow = 1)
  R22 <- R[indices_2, indices_2]

  # Compute the inverse of R22
  R22_inv <- solve(R22)

  # Ensure all variables are numeric matrices
  R12 <- as.matrix(R12)
  R22_inv <- as.matrix(R22_inv)

  # Calculate the conditional variance
  conditional_variance <- R11 - R12 %*% R22_inv %*% t(R12)

  # Calculate the conditional mean for each sample
  coeffs <- R12 %*% R22_inv
  conditional_means <- t(R12 %*% R22_inv %*% t(X2_samples))

  # Generate samples from a Gaussian distribution
  generated_samples <- matrix(NA, nrow = n_samples, ncol = 1)
  for (i in 1:nrow(X2_samples)) {
    generated_samples[i] <- stats::rnorm(1, mean = conditional_means[i],
                                         sd = sqrt(conditional_variance))
  }

  return(list(
    generated_samples = generated_samples,
    conditional_means = conditional_means,
    conditional_variance = conditional_variance,
    coeffs = coeffs
  ))
}


#' Generate Outcome Rank Samples with Marginal Covariate Ranks
#'
#' Constructs a full correlation matrix for the covariates and outcome, and
#' generates outcome rank samples by conditioning on the simulated covariate ranks.
#'
#' Use this function when you have marginal (unconditional) covariate ranks.
#'
#' @param covariate_data A dataframe or matrix containing the simulated covariate
#'   data. Used only to fit the Gaussian copula if \code{gaussian_bn_fit} is
#'   not provided.
#' @param marginal_covariate_ranks A dataframe or matrix containing the simulated
#'   marginal covariate ranks (uniform on [0, 1]).
#' @param vine_cor_params A numeric vector specifying the vine correlation parameters.
#' @param topoOrder Optional numeric vector specifying the topological order of
#'   the vine structure. If not provided, uses \code{1:(n_covariates+1)}.
#' @param gaussian_bn_fit Optional. A pre-fitted DAG-constrained Gaussian BN
#'   (from \code{\link{fit_reference_gaussian_bn}}). If provided, its correlation
#'   matrix \code{R} is used instead of fitting an unconstrained Gaussian copula.
#'
#' @return A list containing:
#' \describe{
#'   \item{gaussianCopulaFit}{The fitted Gaussian copula model (or NULL if using gaussian_bn_fit)}
#'   \item{gaussian_bn_fit}{The DAG-constrained Gaussian BN fit (if used)}
#'   \item{fullCorrelationMatrix}{The full correlation matrix including the outcome}
#'   \item{outcomeRankSamples}{A vector of outcome rank samples on [0, 1]}
#' }
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' cov_data <- matrix(runif(200), ncol = 2)
#' ranks <- cov_data
#' vine_params <- c(0.5, 0.3)
#' results <- simulateMarginalOutcomeSamples(cov_data, ranks, vine_params)
#' }
#'
#' @importFrom stats qnorm pnorm
#' @export
simulateMarginalOutcomeSamples <- function(covariate_data,
                                            marginal_covariate_ranks,
                                            vine_cor_params,
                                            topoOrder = NULL,
                                            gaussian_bn_fit = NULL) {
  cov_data_matrix <- as.matrix(covariate_data)
  n_cov <- ncol(cov_data_matrix)

  if (is.null(topoOrder)) {
    topoOrder <- 1:n_cov  # Only covariates, not outcome
  }

  # Determine correlation matrix: DAG-constrained or unconstrained
  gaussianCopulaFit <- NULL
  if (!is.null(gaussian_bn_fit)) {
    corMatrixMN <- gaussian_bn_fit$R
  } else {
    gaussianCopulaFit <- fitMVGaussianCopula(dataQuantiles = cov_data_matrix, method = 'itau')
    corMatrixMN <- gaussianCopulaFit$correlationMatrix
  }

  # Construct the full correlation matrix
  fullCorrelationMatrix <- computeFullCorMatrix(topoOrder, corMatrixMN, vine_cor_params)

  # Generate outcome rank samples
  X2_samples <- stats::qnorm(as.matrix(marginal_covariate_ranks))
  outcome_model <- multivariate_conditional_mean_and_samples(
    X2_samples = X2_samples,
    R = fullCorrelationMatrix
  )
  outcomeRankSamples <- stats::pnorm(outcome_model$generated_samples)

  return(list(
    gaussianCopulaFit = gaussianCopulaFit,
    gaussian_bn_fit = gaussian_bn_fit,
    fullCorrelationMatrix = fullCorrelationMatrix,
    outcomeRankSamples = outcomeRankSamples
  ))
}


#' Generate Outcome Rank Samples with Conditional Covariate Ranks
#'
#' Constructs a full correlation matrix for the covariates and outcome,
#' transforms conditional ranks to marginal ranks via a Gaussian BN
#' (Route B: SEM propagation), and generates outcome rank samples.
#'
#' Use this function when you have conditional covariate ranks (e.g., from a
#' Bayesian Network where U_{Z2} is really U_{Z2|Z1}).
#'
#' @param covariate_data A dataframe or matrix containing the simulated covariate
#'   data. Used only to auto-fit the Gaussian BN if \code{gaussian_bn_fit} is
#'   not provided.
#' @param cond_covariate_ranks A dataframe or matrix containing the simulated
#'   conditional covariate ranks.
#' @param vine_cor_params A numeric vector specifying the vine correlation parameters.
#' @param topoOrder Optional numeric vector specifying the topological order of
#'   the vine structure. If not provided, uses \code{1:(n_covariates+1)}.
#' @param gaussian_bn_fit Optional. A pre-fitted DAG-constrained Gaussian BN
#'   (from \code{\link{fit_reference_gaussian_bn}} or
#'   \code{\link{fit_gaussian_copula_dag}}). If provided, its correlation
#'   matrix \code{R} is used instead of fitting an unconstrained Gaussian copula.
#' @param parents Optional list. DAG parent structure for each covariate.
#'   Required for \code{\link{uncondition_conditional_ranks}} when using
#'   DAG-constrained R. If \code{gaussian_bn_fit} is provided, defaults to
#'   \code{gaussian_bn_fit$parents}.
#'
#' @return A list containing:
#' \describe{
#'   \item{gaussianCopulaFit}{The fitted Gaussian copula model (or NULL if using gaussian_bn_fit)}
#'   \item{gaussian_bn_fit}{The DAG-constrained Gaussian BN fit (if used)}
#'   \item{fullCorrelationMatrix}{The full correlation matrix including the outcome}
#'   \item{outcomeRankSamples}{A vector of outcome rank samples on [0, 1]}
#' }
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' # Simulate from a BN
#' U1 <- runif(100)
#' Z1 <- qgamma(U1, shape = 2, scale = 2)
#' U2_1 <- runif(100)  # This is U_{Z2|Z1}
#' Z2 <- qgamma(U2_1, shape = 2 + 1.5 * Z1, scale = 1)
#'
#' cov_data <- cbind(Z1, Z2)
#' cond_ranks <- cbind(U1, U2_1)
#' vine_params <- c(0.5, 0.3)
#' dag_parents <- list(integer(0), c(1))
#'
#' # Pre-fit Gaussian BN (Route B)
#' bn_fit <- fit_reference_gaussian_bn(cov_data, dag_parents)
#' results <- simulateConditionalOutcomeSamples(cov_data, cond_ranks,
#'                                               vine_params,
#'                                               gaussian_bn_fit = bn_fit,
#'                                               parents = dag_parents)
#' }
#'
#' @importFrom stats qnorm pnorm
#' @export
simulateConditionalOutcomeSamples <- function(covariate_data,
                                               cond_covariate_ranks,
                                               vine_cor_params,
                                               topoOrder = NULL,
                                               gaussian_bn_fit = NULL,
                                               parents = NULL) {
  cov_data_matrix <- as.matrix(covariate_data)
  n_cov <- ncol(cov_data_matrix)

  if (is.null(topoOrder)) {
    topoOrder <- 1:n_cov  # Only covariates, not outcome
  }

  # Determine correlation matrix: DAG-constrained (Route B) or unconstrained
  gaussianCopulaFit <- NULL
  if (!is.null(gaussian_bn_fit)) {
    # Route B: use pre-fitted DAG-constrained Gaussian BN
    corMatrixMN <- gaussian_bn_fit$R
    if (is.null(parents)) {
      parents <- gaussian_bn_fit$parents
    }
  } else {
    # Auto-fit: if parents are provided, fit DAG-constrained; otherwise unconstrained
    if (!is.null(parents)) {
      gaussian_bn_fit <- fit_reference_gaussian_bn(cov_data_matrix, parents)
      corMatrixMN <- gaussian_bn_fit$R
    } else {
      gaussianCopulaFit <- fitMVGaussianCopula(dataQuantiles = cov_data_matrix, method = 'itau')
      corMatrixMN <- gaussianCopulaFit$correlationMatrix
    }
  }

  # Construct the full correlation matrix including the outcome
  fullCorrelationMatrix <- computeFullCorMatrix(topoOrder, corMatrixMN, vine_cor_params)

  # Transform conditional ranks to marginal ranks
  marginal_covariate_ranks <- uncondition_conditional_ranks(
    cond_covariate_ranks, corMatrixMN, parents = parents
  )

  # Generate outcome rank samples
  X2_samples <- stats::qnorm(as.matrix(marginal_covariate_ranks))
  outcome_model <- multivariate_conditional_mean_and_samples(
    X2_samples = X2_samples,
    R = fullCorrelationMatrix
  )
  outcomeRankSamples <- stats::pnorm(outcome_model$generated_samples)

  return(list(
    gaussianCopulaFit = gaussianCopulaFit,
    gaussian_bn_fit = gaussian_bn_fit,
    fullCorrelationMatrix = fullCorrelationMatrix,
    outcomeRankSamples = outcomeRankSamples
  ))
}
