#' @title Correlation Computation Functions
#' @description Functions for computing full and partial correlation matrices
#'   from vine copula structures.
#' @name correlation
NULL

#' Compute the full correlation matrix from partial correlations
#'
#' This function computes the full correlation matrix from the partial correlations
#' and the topological order of the vine copula structure. It reconstructs marginal
#' correlations from the vine structure.
#'
#' @param topoOrder A vector specifying the topological order of the vine copula structure.
#' @param corMatrixMN The correlation matrix of the M+N variables (covariates only).
#' @param vineCorParams The vine correlation parameters linking covariates to the outcome.
#'
#' @return The full correlation matrix including the outcome variable.
#'
#' @examples
#' \dontrun{
#' topoOrder <- c(2, 1)
#' corMatrixMN <- matrix(c(1, 0.5, 0.5, 1), ncol = 2)
#' vineCorParams <- c(0.6, 0.4)
#' fullCor <- computeFullCorMatrix(topoOrder, corMatrixMN, vineCorParams)
#' }
#'
#' @export
computeFullCorMatrix <- function(topoOrder, corMatrixMN, vineCorParams) {
  D <- length(topoOrder)

  if (!all(dim(corMatrixMN) == c(D, D))) {
    stop("Dimension mismatch: corMatrixMN should match the length of topoOrder")
  }

  # Initialize full correlation matrix with diagonal ones
  fullCorMatrix <- diag(1, D + 1)
  fullCorMatrix[1:D, 1:D] <- corMatrixMN

  # Add the M+Nth variable (last of N) and its marginal correlation with Y
  fullCorMatrix[D, D + 1] <- fullCorMatrix[D + 1, D] <- head(vineCorParams, 1)

  # Define the initial conditioning set B as the last variable
  conditioningSet <- topoOrder[1]

  # Loop over N variables in reverse topological order
  for (i in topoOrder[2:D]) {
    rho <- vineCorParams[which(topoOrder[1:D] == i)]

    # Set A is Y and the current variable
    Sigma_AB <- matrix(fullCorMatrix[c(i, D + 1), c(conditioningSet)], nrow = 2)
    Sigma_BB <- corMatrixMN[conditioningSet, conditioningSet]

    # Compute marginal correlation
    rho_marginal <- computeConditionalCovariance(rho, Sigma_AB, Sigma_BB)

    # Append the computed marginal correlation to the full correlation matrix
    fullCorMatrix[i, D + 1] <- fullCorMatrix[D + 1, i] <- rho_marginal

    # Add the current variable to the conditioning set
    conditioningSet <- c(i, conditioningSet)
  }

  return(fullCorMatrix)
}


#' Compute conditional covariance from partial correlation
#'
#' Internal function to compute the marginal correlation from a partial correlation
#' using the conditional covariance formula.
#'
#' @param rho The partial correlation coefficient.
#' @param Sigma_AB The cross-covariance matrix between variables A and conditioning set B.
#' @param Sigma_BB The covariance matrix of the conditioning set B.
#'
#' @return The marginal correlation.
#'
#' @export
computeConditionalCovariance <- function(rho, Sigma_AB, Sigma_BB) {
  if (length(Sigma_BB) > 1) {
    Sigma_BB_inv <- solve(Sigma_BB)
  } else {
    Sigma_BB_inv <- 1
  }
  Sigma_cond <- Sigma_AB %*% Sigma_BB_inv %*% t(Sigma_AB)

  cond_var_ii <- 1 - diag(Sigma_cond)
  rho_marginal <- sqrt(prod(cond_var_ii)) * rho + Sigma_cond[1, 2]

  return(rho_marginal)
}


#' Compute partial correlations from full correlation matrix
#'
#' This function computes the partial correlations for each variable in the
#' topological order of the vine copula structure by conditioning on previous
#' variables.
#'
#' @param fullCorMatrix The full correlation matrix including the outcome.
#' @param topoOrder A vector specifying the topological order of the vine copula structure.
#'
#' @return A vector of partial correlations.
#'
#' @examples
#' \dontrun{
#' fullCorMatrix <- matrix(c(1, 0.5, 0.3, 0.5, 1, 0.4, 0.3, 0.4, 1), ncol = 3)
#' topoOrder <- c(1, 2)
#' partials <- computePartialCorrelations(fullCorMatrix, topoOrder)
#' }
#'
#' @export
computePartialCorrelations <- function(fullCorMatrix, topoOrder) {
  # Drop Y from topoOrder
  topoOrder <- topoOrder[1:(length(topoOrder) - 1)]

  # Initialize the vector to hold the partial correlations
  partialCorrs <- rep(NA, length(topoOrder))
  Y_idx <- dim(fullCorMatrix)[1]

  # Loop over the variables in the topological order
  for (i in seq_along(topoOrder)) {
    if (i == 1) {
      # If there are no previous variables, the partial correlation is just the correlation
      partialCorrs[i] <- fullCorMatrix[i, Y_idx]
    } else {
      setA <- c(topoOrder[i], length(topoOrder) + 1)
      setB <- topoOrder[1:(i - 1)]

      # Compute the submatrices of the full correlation matrix
      Sigma_AA <- fullCorMatrix[setA, setA]
      Sigma_AB <- fullCorMatrix[setA, setB]
      Sigma_BB <- fullCorMatrix[setB, setB]
      Sigma_BA <- t(Sigma_AB)

      # Compute the conditional covariance matrix Sigma_{A|B}
      Sigma_BB_inv <- solve(Sigma_BB)
      Sigma_A_B <- Sigma_AA - Sigma_AB %*% Sigma_BB_inv %*% Sigma_BA

      # Compute the partial correlation
      partialCorrs[i] <- Sigma_A_B[1, 2] / sqrt(Sigma_A_B[1, 1] * Sigma_A_B[2, 2])
    }
  }

  return(partialCorrs)
}


#' Calculate sequential partial correlations from a correlation matrix
#'
#' Computes partial correlations sequentially, conditioning on all previous
#' variables in the order defined by the matrix columns.
#'
#' @param corMatrix A square correlation matrix.
#'
#' @return A numeric vector of partial correlations.
#'
#' @export
calculateSequentialPartialCorrelations <- function(corMatrix) {
  if (!is.square.matrix(corMatrix)) {
    stop("corMatrix must be a square matrix")
  }

  M <- ncol(corMatrix) - 1
  partialCorrelations <- numeric(M)

  for (i in 1:M) {
    currentSet <- 1:i
    targetVar <- ncol(corMatrix)

    precisionMatrix <- solve(corMatrix[c(currentSet, targetVar), c(currentSet, targetVar)])

    partialCorrelations[i] <- -precisionMatrix[i, length(currentSet) + 1] /
      sqrt(precisionMatrix[i, i] * precisionMatrix[length(currentSet) + 1, length(currentSet) + 1])
  }

  return(partialCorrelations)
}


#' Check if a matrix is square
#'
#' @param mat A matrix.
#' @return Logical indicating whether the matrix is square.
#' @keywords internal
is.square.matrix <- function(mat) {
  nrow(mat) == ncol(mat)
}


#' Compute Standardized Precision Matrix
#'
#' Calculates the standardized empirical precision matrix from simulation data.
#' Normalizes the input data, computes the covariance matrix, inverts it to get
#' the precision matrix, and standardizes to facilitate interpretation of partial
#' correlations within Gaussian-distributed variables.
#'
#' @param sim_quantiles_data A numeric matrix or data frame where each column
#'   represents quantiles from the simulation data. Assumed to be marginally uniform.
#'
#' @return A standardized precision matrix where off-diagonal elements represent
#'   standardized partial correlations and diagonal elements equal 1.
#'
#' @examples
#' \dontrun{
#' data <- matrix(runif(300), ncol = 3)
#' prec_mat <- compute_standardized_precision_matrix(data)
#' }
#'
#' @importFrom stats qnorm cov
#' @export
compute_standardized_precision_matrix <- function(sim_quantiles_data) {
  cov_matrix_np <- stats::cov(stats::qnorm(sim_quantiles_data))

  # Compute standardised precision matrix
  precision_matrix_np <- solve(cov_matrix_np)
  diag_sqrt_inv_np <- 1 / sqrt(diag(precision_matrix_np))
  standardized_precision_matrix_np <- precision_matrix_np * outer(diag_sqrt_inv_np, diag_sqrt_inv_np)

  return(standardized_precision_matrix_np)
}
