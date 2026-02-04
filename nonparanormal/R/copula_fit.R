#' @title Copula Fitting Functions
#' @description Functions for fitting multivariate copula models to data.
#' @name copula_fit
NULL

#' Fit a multivariate Gaussian copula model to data
#'
#' This function fits a multivariate Gaussian copula model to the input data
#' using maximum likelihood estimation (MLE) or inversion of Kendall's tau.
#' It returns the estimated parameters, including the correlation matrix and
#' standard error matrix.
#'
#' @param dataQuantiles A matrix of quantiles of the input data with rows
#'   representing observations and columns representing variables. Values
#'   should be in (0, 1).
#' @param method Character string specifying the fitting method. Default is 'itau'
#'   (inversion of Kendall's tau). Other options include 'mpl' (maximum
#'   pseudo-likelihood) and 'ml' (maximum likelihood).
#'
#' @return A list containing:
#' \describe{
#'   \item{gaussCop}{The Gaussian copula object}
#'   \item{fit}{The fitted copula model object}
#'   \item{correlationMatrix}{The estimated correlation matrix}
#'   \item{stdErrorMatrix}{The standard error matrix for the correlation estimates}
#' }
#'
#' @examples
#' \dontrun{
#' # Generate some uniform data
#' set.seed(123)
#' data <- matrix(runif(300), ncol = 3)
#' result <- fitMVGaussianCopula(data, method = 'itau')
#' print(result$correlationMatrix)
#' }
#'
#' @importFrom copula normalCopula fitCopula
#' @importFrom stats coef vcov
#' @export
fitMVGaussianCopula <- function(dataQuantiles, method = 'itau') {
  if (!requireNamespace("copula", quietly = TRUE)) {
    stop("copula package is not installed. Please install it using install.packages('copula')")
  }

  # Define a Gaussian copula model with an appropriate dimension
  nVars <- ncol(dataQuantiles)
  gaussCop <- copula::normalCopula(dim = nVars, dispstr = "un")

  # Fit the MVG copula model to the data
  fit <- copula::fitCopula(gaussCop, data = dataQuantiles, method = method)

  # Extract the estimated parameters
  param_estimates <- stats::coef(fit)

  # Construct the correlation matrix from the estimates
  corMatrix <- matrix(0, nVars, nVars)
  corMatrix[lower.tri(corMatrix, diag = FALSE)] <- param_estimates
  corMatrix <- corMatrix + t(corMatrix) + diag(nVars)

  # Extract standard errors - use tryCatch as vcov may not always be available
  std_errors <- tryCatch({
    sqrt(diag(stats::vcov(fit)))
  }, error = function(e) {
    rep(NA, length(param_estimates))
  })

  # Construct a matrix for standard errors
  stdErrorMatrix <- matrix(0, nVars, nVars)
  stdErrorMatrix[lower.tri(stdErrorMatrix, diag = FALSE)] <- std_errors
  stdErrorMatrix <- stdErrorMatrix + t(stdErrorMatrix)

  return(list(
    gaussCop = gaussCop,
    fit = fit,
    correlationMatrix = corMatrix,
    stdErrorMatrix = stdErrorMatrix
  ))
}
