library(bnlearn)
library(CondIndTests)
library(copula)
library(GeneralisedCovarianceMeasure)
library(ppcor)
library(VineCopula)

# =============================================================================
# DAG Helper Functions for Longitudinal Models
# =============================================================================

#' Create Longitudinal DAG Structure
#'
#' Generates a DAG parent structure for longitudinal/temporal models with
#' multiple covariates per time point. Column ordering: Z1_1, Z2_1, ..., Z1_2, Z2_2, ...
#'
#' @param n_time Number of time points.
#' @param n_cov Number of covariates per time point.
#' @param structure Type of temporal structure: "markov" (default), "ar1", or "full" (D-vine).
#'
#' @return A list of parent indices with length n_time * n_cov.
#'
#' @examples
#' # 3 time points, 2 covariates, Markov structure
#' parents <- make_longitudinal_dag(n_time = 3, n_cov = 2, structure = "markov")
make_longitudinal_dag <- function(n_time, n_cov, structure = c("markov", "ar1", "full")) {
  structure <- match.arg(structure)

  if (n_time < 1) stop("n_time must be at least 1")
  if (n_cov < 1) stop("n_cov must be at least 1")

  n_vars <- n_time * n_cov
  parents <- vector("list", n_vars)

  # Helper: (covariate d, time t) -> column index (1-based)
  col_idx <- function(d, t) {
    (t - 1) * n_cov + d
  }

  # Generate variable names
  var_names <- character(n_vars)
  for (t in seq_len(n_time)) {
    for (d in seq_len(n_cov)) {
      var_names[col_idx(d, t)] <- paste0("Z", d, "_", t)
    }
  }

  # Build parent structure
  for (t in seq_len(n_time)) {
    for (d in seq_len(n_cov)) {
      j <- col_idx(d, t)

      if (structure == "full") {
        # D-vine: all previous columns
        if (j == 1) {
          parents[[j]] <- integer(0)
        } else {
          parents[[j]] <- seq_len(j - 1)
        }

      } else if (structure == "ar1") {
        # Pure AR(1): only same covariate at previous time
        if (t == 1) {
          parents[[j]] <- integer(0)
        } else {
          parents[[j]] <- col_idx(d, t - 1)
        }

      } else {
        # "markov" structure
        pa <- integer(0)

        # AR term: same covariate at previous time
        if (t > 1) {
          pa <- c(pa, col_idx(d, t - 1))
        }

        # Cross-sectional: all preceding covariates at current time
        if (d > 1) {
          pa <- c(pa, sapply(seq_len(d - 1), function(dd) col_idx(dd, t)))
        }

        parents[[j]] <- sort(pa)
      }
    }
  }

  attr(parents, "var_names") <- var_names
  return(parents)
}

#' Create Simple Chain DAG
#'
#' Creates a DAG: X1 -> X2 -> X3 -> ... -> Xn.
#'
#' @param n_vars Number of variables in the chain.
#' @return A list of parent indices.
make_chain_dag <- function(n_vars) {
  if (n_vars < 1) stop("n_vars must be at least 1")

  parents <- vector("list", n_vars)
  parents[[1]] <- integer(0)

  if (n_vars > 1) {
    for (j in 2:n_vars) {
      parents[[j]] <- j - 1L
    }
  }

  return(parents)
}

#' Simulate data from an R-vine copula model.
#' 
#' This function generates synthetic data from an R-vine copula model specified by the given structure,
#' family, and parameter matrices.
#' 
#' @param structureMatrix A vector or matrix specifying the structure of the R-vine copula model.
#'                        The structure matrix encodes the dependence structure among variables.
#' @param familyMatrix A vector or matrix specifying the families of bivariate copulas used in the R-vine model.
#'                     Each entry in the matrix corresponds to a pair of variables and specifies the copula family.
#' @param parameterMatrix A vector or matrix specifying the parameters of the bivariate copulas used in the R-vine model.
#'                        Each entry in the matrix corresponds to a pair of variables and specifies the copula parameters.
#' @param sampleSize The number of observations to simulate.
#' @return A matrix containing the simulated data with rows representing observations and columns representing variables.
#' @examples
#' simulateRVineData(structureMatrix = c(1, 2, 0, 1), 
#'                   familyMatrix = c(1, 3, 0, 1), 
#'                   parameterMatrix = c(0.5, 0.8, 0), 
#'                   sampleSize = 100)
simulateRVineData <- function(structureMatrix, familyMatrix, parameterMatrix, sampleSize = 300, seed=123) {
  # Ensure matrices are in the correct format
  # structureMatrix <- matrix(structureMatrix, ncol = sqrt(length(structureMatrix)))
  # familyMatrix <- matrix(familyMatrix, ncol = sqrt(length(familyMatrix)))
  # parameterMatrix <- matrix(parameterMatrix, ncol = sqrt(length(parameterMatrix)))
  
  # Define variable names dynamically based on the structure matrix's dimension
  varNames <- paste0("V", 1:(max(structureMatrix[structureMatrix > 0])))
  
  # Define the RVineMatrix object
  RVM <- RVineMatrix(Matrix = structureMatrix, family = familyMatrix, par = parameterMatrix, names = varNames)
  
  # Set seed for reproducibility (optional)
  set.seed(seed)
  
  # Simulate data from the defined R-vine model
  simdata <- RVineSim(sampleSize, RVM)
  
  return(list(RVM=RVM, simdata=simdata))
}

#' Fit a multivariate Gaussian copula model to data.
#' 
#' This function fits a multivariate Gaussian copula model to the input data using maximum likelihood estimation (MLE).
#' It returns the estimated parameters, including the correlation matrix and standard error matrix.
#' 
#' @param dataQuantiles A matrix of quantiles of the input data with rows representing observations and columns representing variables.
#' @return A list containing the fitted copula model object, estimated correlation matrix, and standard error matrix.
#' @examples
#' fitMVGaussianCopula(dataQuantiles = my_data)
fitMVGaussianCopula <- function(dataQuantiles, method='itau') {
  # Ensure the copula package is loaded
  if (!requireNamespace("copula", quietly = TRUE)) {
    stop("copula package is not installed. Please install it using install.packages('copula')")
  }
  
  # Define a Gaussian copula model with an appropriate dimension
  nVars <- ncol(dataQuantiles)  # Number of variables
  gaussCop <- normalCopula(dim = nVars, dispstr = "un")
  
  # Fit the MVG copula model to the data using MLE
  fit <- fitCopula(gaussCop, data = dataQuantiles, method = method)
  
  # Extract the estimated parameters
  param_estimates <- coef(fit)
  
  # Construct the correlation matrix from the estimates
  corMatrix <- matrix(0, nVars, nVars)
  corMatrix[lower.tri(corMatrix, diag = FALSE)] <- param_estimates
  corMatrix <- corMatrix + t(corMatrix) + diag(nVars)  # Make symmetric and add 1s on the diagonal
  
  # Extract standard errors
  std_errors <- sqrt(diag(vcov(fit)))
  
  # Construct a matrix for standard errors
  stdErrorMatrix <- matrix(0, nVars, nVars)
  stdErrorMatrix[lower.tri(stdErrorMatrix, diag = FALSE)] <- std_errors
  stdErrorMatrix <- stdErrorMatrix + t(stdErrorMatrix)  # Make symmetric
  
  # Return a list containing the fit object, correlation matrix, and standard error matrix
  return(list(gaussCop=gaussCop, fit = fit, correlationMatrix = corMatrix, stdErrorMatrix = stdErrorMatrix))
}

#' Compute the full correlation matrix from partial correlations.
#' 
#' This function computes the full correlation matrix from the partial correlations and the topological order of the vine copula structure.
#' 
#' @param topoOrder A vector specifying the topological order of the vine copula structure.
#' @param corMatrixMN The correlation matrix of the M+N variables.
#' @param vineCorParams The parameters of the vine copula model.
#' @return The full correlation matrix.
#' @examples
#' computeFullCorMatrix(topoOrder = c(1, 2, 3), corMatrixMN = my_cor_matrix, vineCorParams = my_params)
computeFullCorMatrix <- function(topoOrder, corMatrixMN, vineCorParams) {
  # Ensure the input correlation matrix matches the topological order size
  D <- length(topoOrder)
  # N <- length(vineCorParams)
  # M <- D - N
  if (!all(dim(corMatrixMN) == c(D, D))) {
    stop("Dimension mismatch: corMatrixMN should match the length of topoOrder")
  }
  
  # Initialize full correlation matrix with diagonal ones
  fullCorMatrix <- diag(1, D+1)
  fullCorMatrix[1:D, 1:D] <- corMatrixMN
  
  # Add the M+Nth variable (last of N) and its marginal correlation with Y
  fullCorMatrix[D, D + 1] <- fullCorMatrix[D + 1, D] <- head(vineCorParams, 1)
  
  # Define the initial conditioning set B as the last variable
  conditioningSet <- topoOrder[1]
  
  # Loop over N variables in reverse topological order
  for (i in topoOrder[2:D]) {
    # Need to fix the line below to ensure consistency in the order we're sampling variables.
    rho <- vineCorParams[which(topoOrder[1:D] == i)]
    # if (i %in% topoOrder[(M+1):D]) {
    #   # Set A is Y and the current variable
    #   rho <- vineCorParams[which(topoOrder[(D-N+1):D] == i)]
    # } else {
    #   rho <- 0 
    # }
    # Set A is Y and the current variable
    Sigma_AB <- matrix(fullCorMatrix[c(i, D+1), c(conditioningSet)], nrow=2)
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

#' Compute the full correlation matrix from partial correlations.
#' 
#' This function computes the full correlation matrix from the partial correlations and the topological order of the vine copula structure.
#' 
#' @param topoOrder A vector specifying the topological order of the vine copula structure.
#' @param corMatrixMN The correlation matrix of the M+N variables.
#' @param vineCorParams The parameters of the vine copula model.
#' @return The full correlation matrix.
#' @examples
#' computeFullCorMatrix(topoOrder = c(1, 2, 3), corMatrixMN = my_cor_matrix, vineCorParams = my_params)
computeConditionalCovariance <- function(rho, Sigma_AB, Sigma_BB) {
  # if (!all(dim(Sigma_AB) == c(2, length(Sigma_BB)))) {
  #   stop("Dimension mismatch: Sigma_AB should be 2xlength(Sigma_BB)")
  # }
  if (length(Sigma_BB) > 1){
    Sigma_BB_inv <- solve(Sigma_BB)
  } else {
    Sigma_BB_inv <- 1
  }
  Sigma_cond <- Sigma_AB %*% Sigma_BB_inv %*% t(Sigma_AB)
  
  cond_var_ii <- 1 - diag(Sigma_cond)
  rho_marginal <- sqrt(prod(cond_var_ii)) * rho + Sigma_cond[1,2]
  
  return(rho_marginal)
}

#' Compute partial correlations for each variable.
#' 
#' This function computes the partial correlations for each variable in the topological order of the vine copula structure.
#' 
#' @param fullCorMatrix The full correlation matrix.
#' @param topoOrder A vector specifying the topological order of the vine copula structure.
#' @return A vector of partial correlations.
#' @examples
#' computePartialCorrelations(fullCorMatrix = my_full_cor_matrix, topoOrder = c(1, 2, 3))
computePartialCorrelations <- function(fullCorMatrix, topoOrder) {
  # Drop Y from topoOrder
  topoOrder <- topoOrder[1:(length(topoOrder)-1)]
  
  # Initialize the vector to hold the partial correlations
  partialCorrs <- rep(NA, length(topoOrder))
  Y_idx <- dim(fullCorMatrix)[1]
  
  # Loop over the variables in the topological order
  for (i in seq_along(topoOrder)) {
    # Set A includes the current variable and Y
    if (i == 1) {
      # If there are no previous variables, the partial correlation is just the correlation
      partialCorrs[i] <- fullCorMatrix[i, Y_idx]
    } else {
      setA <- c(topoOrder[i], length(topoOrder)+1)
      
      # Set B includes all previous variables
      setB <- topoOrder[1:(i-1)]
      
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
  
  # Return the vector of partial correlations
  return(partialCorrs)
}


calculateSequentialPartialCorrelations <- function(corMatrix) {
  # Ensure corMatrix is square
  if (!is.square.matrix(corMatrix)) {
    stop("corMatrix must be a square matrix")
  }
  
  # Number of variables (excluding the last one for partial correlation calculation)
  M <- ncol(corMatrix) - 1
  
  # Initialize vector to store the partial correlations
  partialCorrelations <- numeric(M)
  
  # Loop through the first M-1 variables
  for (i in 1:M) {
    # Define the current set of variables (i and all previous ones) and the target variable
    currentSet <- 1:i
    targetVar <- ncol(corMatrix)
    
    # Compute the precision matrix (inverse of the correlation matrix) for the current set + target
    precisionMatrix <- solve(corMatrix[c(currentSet, targetVar), c(currentSet, targetVar)])
    
    # Calculate the partial correlation for the ith variable with the target,
    # conditioned on all previous variables in the set
    partialCorrelations[i] <- -precisionMatrix[i, length(currentSet)+1] / sqrt(precisionMatrix[i, i] * precisionMatrix[length(currentSet)+1, length(currentSet)+1])
  }
  
  return(partialCorrelations)
}

# Helper function to check if a matrix is square
is.square.matrix <- function(mat) {
  nrow(mat) == ncol(mat)
}

#' Compute Standardized Precision Matrix
#'
#' This function calculates the standardized empirical precision matrix from a given simulation data set.
#' It is designed to normalize the input data, compute the covariance matrix, invert it to get the precision matrix,
#' and then standardize this precision matrix to facilitate the interpretation of partial correlations within
#' Gaussian-distributed variables.
#'
#' @param sim_quantiles_data A numeric matrix or data frame where each column represents a set of quantiles from the simulation data.
#'                           It is assumed that the data are marginally uniform.
#'
#' @return A standardized precision matrix, which is a square matrix where off-diagonal elements represent
#'         standardized partial correlations and diagonal elements are equal to 1.
#'         
#' @examples
#' # Example usage:
#' standardized_precision_matrix <- compute_standardized_precision_matrix(sim_quantiles_data)
#' 
#' @export
compute_standardized_precision_matrix <- function(sim_quantiles_data) {
  cov_matrix_np <- cov(qnorm(sim_quantiles_data))
  
  # Compute standardised precision matrix
  precision_matrix_np <- solve(cov_matrix_np)
  diag_sqrt_inv_np <- 1 / sqrt(diag(precision_matrix_np))
  standardized_precision_matrix_np <- precision_matrix_np * outer(diag_sqrt_inv_np, diag_sqrt_inv_np)
  
  # Return the standardized precision matrix
  return(standardized_precision_matrix_np)
}

#' Reparameterize Vine Matrices for Non-Paranormal Approximation
#'
#' This function updates vine copula matrices (`structureMatrix`, `familyMatrix`, `parameterMatrix`)
#' according to a specific set of transformation rules, facilitating the approximation of a non-paranormal
#' (or Gaussian copula) model. It is designed to manipulate these matrices to align with the non-paranormal
#' approximation, which allows for a more flexible modeling of dependencies than the standard normal
#' distribution by applying transformations to the data. The function adjusts the structure, family, and
#' parameter matrices based on the provided partial correlations (`partialCors`), effectively reparameterizing
#' the vine.
#'
#' @param structureMatrix A square, lower triangular matrix representing the R-vine structure, to be adjusted
#'        according to the non-paranormal approximation requirements.
#' @param familyMatrix A square, lower triangular matrix representing the family of the vine copulae, reflecting
#'        the types of dependencies among variables.
#' @param parameterMatrix A square, lower triangular matrix representing the parameters of the vine copulae,
#'        which will be adjusted to align with the non-paranormal approximation.
#' @param partialCors A numeric vector of size D-1, where D is the dimension of the matrices, representing
#'        partial correlations. These are used to update the parameter matrix to reflect the non-paranormal
#'        approximation.
#'
#' @return A list containing three elements: `newStructureMatrix`, `newFamilyMatrix`, and `newParameterMatrix`.
#'         Each matrix is updated to facilitate the modeling of dependencies according to the non-paranormal
#'         approximation.
#'
#' @examples
#' # Define example matrices (for simplicity, D = 4)
#' structureMatrix <- matrix(c(4, 0, 0, 0, 3, 4, 0, 0, 2, 3, 4, 0, 1, 2, 3, 4), nrow = 4, byrow = TRUE)
#' familyMatrix <- structureMatrix # In practice, they would differ
#' parameterMatrix <- structureMatrix * 0.5 # Normally, specific copula parameters
#' partialCors <- c(0.1, 0.2, 0.3)
#'
#' # Reparameterize the matrices
#' results <- updateVineMatrices(structureMatrix, familyMatrix, parameterMatrix, partialCors)
#'
#' # Access the updated matrices for further analysis or modeling
#' newStructureMatrix <- results$newStructureMatrix
#' newFamilyMatrix <- results$newFamilyMatrix
#' newParameterMatrix <- results$newParameterMatrix
#'
#' @export
updateVineMatrices <- function(structureMatrix, familyMatrix, parameterMatrix, partialCors) {
  # Determine the dimension D of the matrices
  D <- nrow(structureMatrix)
  
  # Validate input dimensions and types
  if (D != ncol(structureMatrix) || D != nrow(familyMatrix) || D != ncol(familyMatrix) || D != nrow(parameterMatrix) || D != ncol(parameterMatrix)) {
    stop("All input matrices must be square and of the same dimension.")
  }
  if (length(partialCors) != D - 1) {
    stop("partialCors must have length D-1, where D is the dimension of the matrices.")
  }
  
  # Update structureMatrix
  newStructureMatrix <- (structureMatrix - 1)
  newStructureMatrix[1,1] <- newStructureMatrix[1,1] + D
  newStructureMatrix[upper.tri(newStructureMatrix)] <- 0
  
  # Update familyMatrix
  newFamilyMatrix <- familyMatrix
  newFamilyMatrix[,2:(D-1)] <- familyMatrix[,1:(D-2)]
  newFamilyMatrix[(2:D),1] <- 1
  
  # Update parameterMatrix
  newParameterMatrix <- parameterMatrix
  newParameterMatrix[D, 2:(D-1)] <- parameterMatrix[D, 1:(D-2)]
  newParameterMatrix[(D:2), 1] <- partialCors
  
  # Return the updated matrices as a list
  return(list(StructureMatrix = newStructureMatrix, 
              FamilyMatrix = newFamilyMatrix, 
              ParameterMatrix = newParameterMatrix))
}

#' Simulate and Reparameterize Vine Copula Data
#'
#' Simulates data from a specified vine copula model, fits a multivariate Gaussian copula to the simulated data,
#' reparameterizes the vine copula according to the non-paranormal approximation using partial correlations,
#' and simulates new data from the reparameterized vine. This process is designed to align the copula model
#' closer to the empirical correlations observed in the data, especially for complex dependencies.
#'
#' @param structureMatrix Square, lower triangular matrix specifying the R-vine structure.
#' @param familyMatrix Square, lower triangular matrix specifying the family of the vine copulae.
#' @param parameterMatrix Square, lower triangular matrix specifying the parameters of the vine copulae.
#' @param sampleSize Integer, the number of samples to draw from the vine.
#' @param topoOrder Numeric vector, the topological order of the vine copula.
#' @param vineCorParams Numeric vector, the correlation parameters between pretreatment confounders and Y.
#'
#' @return A list containing `oldVineOutput`, `newVineOutput`, and `standardized_precision_matrix_np`.
#'         - `oldVineOutput`: The simulated data from the original vine specification.
#'         - `newVineOutput`: The simulated data from the reparameterized vine.
#'         - `standardized_precision_matrix_np`: The standardized precision matrix of the new simulated data.
#'         
#' @examples
#' D <- 6
#' sampleSize <- 5000
#' topoOrder <- 1:D
#' vineCorParams <- c(0.5)
#' # Define structureMatrix, familyMatrix, and parameterMatrix as specified above.
#'
#' results <- simulateAndReparameterizeVine(structureMatrix, familyMatrix, parameterMatrix, sampleSize, topoOrder, vineCorParams)
#' 
#' @export
simulateAndReparameterizeVine <- function(structureMatrix, familyMatrix, parameterMatrix, sampleSize, topoOrder, vineCorParams, seed=1) {
  library(copula)
  library(VineCopula)
  
  # Initial vine copula simulation
  oldVineOutput <- simulateRVineData(structureMatrix, familyMatrix, parameterMatrix, sampleSize, seed)
  
  # Fit a multivariate Gaussian copula to the simulated data
  mvgFit <- fitMVGaussianCopula(oldVineOutput$simdata, method = 'itau')
  
  # Reparameterization settings
  corMatrixMN <- mvgFit$correlationMatrix[1:(D-1), 1:(D-1)]
  fullCorMatrixMN <- computeFullCorMatrix(topoOrder, corMatrixMN, vineCorParams)
  partialCors <- computePartialCorrelations(fullCorMatrixMN, topoOrder)
  
  # Update vine matrices based on the non-paranormal approximation
  updatedVine <- updateVineMatrices(structureMatrix, familyMatrix, parameterMatrix, partialCors)
  
  # Simulate data from the updated vine
  newVineOutput <- simulateRVineData(
    updatedVine$StructureMatrix, 
    updatedVine$FamilyMatrix, 
    updatedVine$ParameterMatrix,
    sampleSize,
    seed=seed
  )
  
  # Compute the standardized precision matrix for the new simulated data
  standardized_precision_matrix_np <- compute_standardized_precision_matrix(newVineOutput$simdata)
  
  return(list(
    oldVineOutput = oldVineOutput, 
    newVineOutput = newVineOutput, 
    standardized_precision_matrix_np = standardized_precision_matrix_np, 
    fullCorMatrixMN=fullCorMatrixMN
  ))
}

simulateAndPlot <- function(structureMatrix, familyMatrix, sampleSize, topoOrder, normal_corr_values, general_dep, general_family, seed=1) {
  plot_list <- list() # Initialize an empty list to store ggplot objects
  D <- dim(structureMatrix)[1]
  set.seed(seed)
  for (normal_corr in normal_corr_values) {
    # Update the normal correlation parameter in the parameter matrix
    vineCorParams <- normal_corr
    
    # Run simulation and reparameterization
    npReparamVine <- simulateAndReparameterizeVine(
      structureMatrix, 
      familyMatrix, 
      parameterMatrix, 
      sampleSize, 
      topoOrder, 
      vineCorParams
    )
    
    oldVineOutput <- npReparamVine$oldVineOutput
    # newVineOutput <- npReparamVine$newVineOutput
    ###########################
    ############ HACKY SOLUTION
    ###########################
    newVineOutput <- npReparamVine$newVineOutput
    
    # Final variable is the outcome
    newSimData <- newVineOutput$simdata
    covariate_ranks <- newSimData[, 1:(dim(newSimData)[2] - 1)]
    outcome_model <-multivariate_conditional_mean_and_samples(
      X2_samples = qnorm(covariate_ranks), 
      R = npReparamVine$fullCorMatrixMN
    )
    outcome_quantile_samples <- pnorm(outcome_model$generated_samples)
    newSimData[, dim(newSimData)[2]] <- outcome_quantile_samples
    npReparamVine$newVineOutput$simdata <- newSimData
    newVineOutput$simdata <- newSimData
    ###########################
    ###########################
    ###########################
    
    margins <- oldVineOutput$simdata
    margins_np <- newVineOutput$simdata
    
    # Calculate H-functions for both old and new vine outputs
    F2_3_np <- BiCopHfunc(margins_np[,2], margins_np[,3], family=general_family, par=general_dep)$hfunc2
    F4_3_np <- BiCopHfunc(margins_np[,4], margins_np[,3], family=1, par=normal_corr)$hfunc2
    
    F2_3 <- BiCopHfunc(margins[,2], margins[,3], family=general_family, par=general_dep)$hfunc2
    F4_3 <- BiCopHfunc(margins[,4], margins[,3], family=1, par=normal_corr)$hfunc2
    
    # Generate plots
    p1 <- ggplot(data.frame(F2_3 = F2_3_np, F4_3 = F4_3_np), aes(x = F2_3, y = F4_3)) +
      stat_density_2d(aes(fill = ..level..), geom = "polygon") +  # Use stat_density_2d for filled contours
      scale_fill_viridis_c() +  # Adds a color gradient based on density levels
      labs(x = "F2_3", y = "F4_3", title = paste("NP Contour Plot of F2_3 vs F4_3 (corr =", normal_corr, ")")) +
      theme_minimal()
    
    
    p2 <- ggplot(data.frame(F2_3 = F2_3, F4_3 = F4_3), aes(x = F2_3, y = F4_3)) +
      stat_density_2d(aes(fill = ..level..), geom = "polygon") +  # Use stat_density_2d for filled contours
      scale_fill_viridis_c() +  # Adds a color gradient based on density levels
      labs(x = "F2_3", y = "F4_3", title = paste("True Contour Plot of F2_3 vs F4_3 (corr =", normal_corr, ")")) +
      theme_minimal()
    
    # Add plots to the list
    plot_list[[length(plot_list) + 1]] <- p1
    plot_list[[length(plot_list) + 1]] <- p2
  }
  
  # Combine all plots into a grid
  do.call(grid.arrange, c(plot_list, ncol = 2))
}


multivariate_conditional_mean_and_samples <- function(X2_samples, R) {
  # Determine the dimensions
  k <- ncol(X2_samples)       # Number of variables being conditioned on (X2)
  n_samples <- nrow(X2_samples)
  d <- nrow(R)                # Total number of variables
  
  X2_samples <- as.matrix(X2_samples)
  
  print(paste0("Conditioning the distribution on index "), d)
  # Indices for partitioning the covariance matrix
  indices_1 <- d              # First variable (X1)
  indices_2 <- 1:(d-1)            # Remaining variables (X2)
  
  # Extract the relevant components from the covariance matrix R
  R11 <- R[indices_1, indices_1]         # Variance of X1
  R12 <- matrix(R[indices_1, indices_2], nrow=1)         # Covariance between X1 and X2
  R22 <- R[indices_2, indices_2]         # Covariance matrix of X2
  
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
  
  # Generate samples from a Gaussian distribution with the computed conditional mean and variance
  generated_samples <- matrix(NA, nrow = n_samples, ncol = 1)
  for (i in 1:nrow(X2_samples)) {
    generated_samples[i] <- rnorm(1, mean = conditional_means[i], sd = sqrt(conditional_variance))
  }
  
  # Return the conditional means, conditional variance, and generated samples
  return(list(
    generated_samples = generated_samples,
    conditional_means = conditional_means,
    conditional_variance = conditional_variance,
    coeffs = coeffs
  ))
}

#' Uncondition Conditional Ranks Using a Gaussian Copula Translation
#'
#' Given a matrix (or dataframe) of conditional ranks, this function iterates over the variables and
#' "unconditions" the conditional ranks to obtain their marginal alternatives on the Gaussian (normal quantile) scale.
#'
#' By default, assumes a fully-connected D-vine structure where each variable is conditioned on all
#' previous variables. For sparser structures (e.g., from a DAG), use the parents argument.
#'
#' @param cond_ranks A matrix (or dataframe) of conditional ranks. The first column is assumed to be
#'        unconditional (U_{Z1}), the second column is U_{Z2|Z1}, the third is U_{Z3|Z1,Z2}, etc.
#' @param R A correlation matrix corresponding to the full Gaussian copula that models the joint distribution.
#' @param parents Optional list specifying the parent columns for each variable.
#'   parents[[j]] should be an integer vector of column indices that are parents of variable j.
#'   If NULL (default), assumes D-vine structure (each variable conditioned on all previous).
#'   Use make_longitudinal_dag() to generate appropriate parent structures.
#' @param check_order Logical. If TRUE (default), validates topological order.
#'
#' @return A matrix of the same dimensions as \code{cond_ranks} containing the "unconditioned" values
#'         transformed back to the probability scale [0, 1].
#'
#' @examples
#' # Suppose we have a 3-variable example:
#' cond_ranks <- matrix(c(runif(5), runif(5), runif(5)), ncol = 3)
#' R <- matrix(c(1, 0.5, 0.3,
#'               0.5, 1, 0.4,
#'               0.3, 0.4, 1), nrow = 3, byrow = TRUE)
#' marginal_values <- uncondition_conditional_ranks(cond_ranks, R)
#'
#' # With DAG structure Z1 -> Z2 -> Z3 (Z3 only depends on Z2)
#' parents <- list(integer(0), c(1), c(2))
#' marginal_values <- uncondition_conditional_ranks(cond_ranks, R, parents)
uncondition_conditional_ranks <- function(cond_ranks, R, parents = NULL, check_order = TRUE) {
  # Ensure cond_ranks is a matrix
  cond_ranks <- as.matrix(cond_ranks)
  n <- nrow(cond_ranks)
  d <- ncol(cond_ranks)

  # If parents not specified, use default D-vine structure (all previous)
  if (is.null(parents)) {
    parents <- vector("list", d)
    parents[[1]] <- integer(0)
    if (d > 1) {
      for (j in 2:d) {
        parents[[j]] <- 1:(j - 1)
      }
    }
  }

  # Validate parents structure
  if (length(parents) != d) {
    stop("Length of 'parents' must equal number of columns in cond_ranks")
  }

  # Validate topological order if requested
  if (check_order) {
    for (j in seq_len(d)) {
      pa_j <- parents[[j]]
      if (length(pa_j) > 0) {
        if (any(pa_j >= j)) {
          bad_parents <- pa_j[pa_j >= j]
          stop(sprintf(
            "Invalid topological order: variable %d has parent(s) %s with index >= %d. ",
            j, paste(bad_parents, collapse = ", "), j
          ))
        }
        if (any(pa_j < 1)) {
          stop(sprintf("Invalid parent index for variable %d: indices must be >= 1", j))
        }
      }
    }
  }

  # Cache for R_sub inversions - avoid redundant solve() calls
  inversion_cache <- new.env(hash = TRUE, parent = emptyenv())

  get_R_sub_inv <- function(pa_idx) {
    if (length(pa_idx) == 0) return(NULL)
    pa_key <- paste(sort(pa_idx), collapse = ",")
    if (!exists(pa_key, envir = inversion_cache)) {
      R_sub <- R[pa_idx, pa_idx, drop = FALSE]
      inversion_cache[[pa_key]] <- solve(R_sub)
    }
    return(inversion_cache[[pa_key]])
  }

  # Pre-compute all necessary inversions
  for (j in seq_len(d)) {
    if (length(parents[[j]]) > 0) {
      get_R_sub_inv(parents[[j]])
    }
  }

  # Prepare a matrix to store the marginal (unconditioned) values.
  marginal_values <- matrix(NA_real_, n, d)

  # Process each observation (each row) individually.
  for (i in 1:n) {
    x <- numeric(d)

    for (j in 1:d) {
      pa_j <- parents[[j]]

      if (length(pa_j) == 0) {
        # Root variable: unconditional
        x[j] <- qnorm(cond_ranks[i, j])
      } else {
        # Non-root: uncondition using the specified parents
        r_vec <- matrix(R[j, pa_j], nrow = 1)
        R_sub_inv <- get_R_sub_inv(pa_j)
        x_pa <- matrix(x[pa_j], ncol = 1)

        # Compute the conditional mean.
        mu_j <- as.numeric(r_vec %*% R_sub_inv %*% x_pa)
        # Compute the conditional variance.
        sigma2_j <- 1 - as.numeric(r_vec %*% R_sub_inv %*% t(r_vec))
        sigma_j <- sqrt(max(sigma2_j, 1e-10))

        # "Uncondition" the j-th variable
        x[j] <- mu_j + sigma_j * qnorm(cond_ranks[i, j])
      }
    }
    marginal_values[i, ] <- x
  }

  return(pnorm(marginal_values))
}


# =============================================================================
# DAG-Constrained Gaussian Copula (Route B)
# =============================================================================

#' Normalise Parents List to Integer Indices
#' @keywords internal
normalise_parents <- function(parents, var_names) {
  d <- length(var_names)
  if (!is.null(names(parents))) {
    if (!setequal(names(parents), var_names)) {
      stop("Parent names do not match variable names")
    }
    parents <- parents[var_names]
  } else {
    if (length(parents) != d) {
      stop(sprintf("Length of parents (%d) must equal number of variables (%d)",
                   length(parents), d))
    }
  }
  parents_idx <- lapply(seq_along(parents), function(j) {
    pa <- parents[[j]]
    if (length(pa) == 0 || (length(pa) == 1 && (is.na(pa) || pa == ""))) {
      return(integer(0))
    }
    if (is.character(pa)) {
      idx <- match(pa, var_names)
      if (any(is.na(idx))) stop(sprintf("Unknown parent name(s) for node %d", j))
      return(idx)
    } else if (is.numeric(pa)) {
      pa <- as.integer(pa)
      if (any(pa < 1 | pa > d)) stop(sprintf("Parent index out of bounds for node %d", j))
      return(pa)
    } else {
      stop("parents must be character names or integer indices")
    }
  })
  names(parents_idx) <- var_names
  return(parents_idx)
}

#' Topological Sort Using Kahn's Algorithm
#' @keywords internal
topological_sort_kahn <- function(parents) {
  d <- length(parents)
  children <- vector("list", d)
  for (j in seq_len(d)) children[[j]] <- integer(0)
  for (j in seq_len(d)) {
    for (pa in parents[[j]]) children[[pa]] <- c(children[[pa]], j)
  }
  in_degree <- lengths(parents)
  queue <- which(in_degree == 0)
  order <- integer(0)
  while (length(queue) > 0) {
    node <- queue[1]
    queue <- queue[-1]
    order <- c(order, node)
    for (child in children[[node]]) {
      in_degree[child] <- in_degree[child] - 1
      if (in_degree[child] == 0) queue <- c(queue, child)
    }
  }
  if (length(order) != d) stop("Cycle detected in DAG")
  return(order)
}

#' Fit Gaussian Copula with DAG Structure
#'
#' Projects onto DAG-constrained Gaussian BN family via OLS regression.
#'
#' @param U Matrix of pseudo-observations (n x d) in (0, 1)
#' @param parents List mapping variable index -> parent indices
#' @param topo_order Optional integer vector specifying topological order
#' @param centre Logical: centre Z columns (default TRUE)
#' @param eps Floor for sigma^2_j (default: 1e-10)
#' @return List with B, sigma2, Sigma, R, L, parents, etc.
fit_gaussian_copula_dag <- function(U, parents, topo_order = NULL,
                                     centre = TRUE, eps = 1e-10) {
  U <- as.matrix(U)
  n <- nrow(U)
  d <- ncol(U)
  var_names <- colnames(U)
  if (is.null(var_names)) {
    var_names <- paste0("V", seq_len(d))
    colnames(U) <- var_names
  }
  parents_idx <- normalise_parents(parents, var_names)
  if (length(parents_idx) != d) {
    stop(sprintf("Length of parents (%d) must equal number of columns (%d)",
                 length(parents_idx), d))
  }
  if (is.null(topo_order)) {
    topo_order <- topological_sort_kahn(parents_idx)
  }
  Z <- qnorm(U)
  if (centre) {
    Z <- scale(Z, center = TRUE, scale = FALSE)
  } else {
    stop("centre = FALSE not supported")
  }
  B <- matrix(0, d, d)
  rownames(B) <- colnames(B) <- var_names
  sigma2 <- numeric(d)
  names(sigma2) <- var_names
  regressions <- vector("list", d)
  names(regressions) <- var_names
  for (j in topo_order) {
    pa_j <- parents_idx[[j]]
    if (length(pa_j) == 0) {
      sigma2[j] <- max(var(Z[, j]), eps)
      regressions[[j]] <- list(node = var_names[j], parents = character(0),
                                coefficients = NULL, residual_var = sigma2[j], is_root = TRUE)
    } else {
      fit_lm <- lm(Z[, j] ~ Z[, pa_j, drop = FALSE] - 1)
      beta_j <- coef(fit_lm)
      B[pa_j, j] <- beta_j
      sigma2[j] <- max(var(residuals(fit_lm)), eps)
      regressions[[j]] <- list(node = var_names[j], parents = var_names[pa_j],
                                coefficients = setNames(beta_j, var_names[pa_j]),
                                residual_var = sigma2[j], is_root = FALSE)
    }
  }
  T_mat <- diag(d) - t(B)
  rownames(T_mat) <- colnames(T_mat) <- var_names
  D <- diag(sigma2, nrow = d)
  rownames(D) <- colnames(D) <- var_names
  T_inv <- solve(T_mat)
  Sigma <- T_inv %*% D %*% t(T_inv)
  Sigma <- (Sigma + t(Sigma)) / 2
  rownames(Sigma) <- colnames(Sigma) <- var_names
  R <- cov2cor(Sigma)
  L <- tryCatch({
    t(chol(R))
  }, error = function(e) {
    warning("chol(R) failed; adding small jitter to diagonal")
    t(chol(R + diag(eps, d)))
  })
  rownames(L) <- colnames(L) <- var_names
  return(list(B = B, sigma2 = sigma2, T_mat = T_mat, D = D,
              Sigma = Sigma, R = R, L = L, regressions = regressions,
              topo_order = topo_order, var_names = var_names,
              parents = parents_idx, d = d))
}

#' Fit Reference Gaussian BN from Raw Covariate Data
#'
#' @param Z_ref Matrix of observed covariate values (n x d)
#' @param parents List mapping variable index -> parent indices
#' @param n_ref Integer. If generate_fn provided, number of reference samples
#' @param generate_fn Optional function that generates reference data
#' @param ... Additional arguments passed to fit_gaussian_copula_dag
#' @return Fitted model list from fit_gaussian_copula_dag
fit_reference_gaussian_bn <- function(Z_ref, parents, n_ref = NULL,
                                       generate_fn = NULL, ...) {
  if (!is.null(generate_fn) && !is.null(n_ref)) {
    Z_ref <- generate_fn(n_ref)
  }
  if (is.null(Z_ref)) {
    stop("Either Z_ref must be provided or both generate_fn and n_ref must be specified")
  }
  Z_ref <- as.matrix(Z_ref)
  n <- nrow(Z_ref)
  U_ref <- apply(Z_ref, 2, function(col) {
    rank(col, ties.method = "average") / (n + 1)
  })
  if (!is.null(colnames(Z_ref))) colnames(U_ref) <- colnames(Z_ref)
  fit <- fit_gaussian_copula_dag(U_ref, parents, ...)
  return(fit)
}


#' Generate Outcome Rank Samples from Vine Copula Simulation with Marginal Ranks
#'
#' This function fits a Gaussian copula to the covariate data, constructs a full correlation
#' matrix for the covariates and outcome (using a provided vine correlation parameter and topological order),
#' and then generates outcome rank samples by conditioning on the simulated covariate ranks.
#'
#' @param covariate_data A dataframe or matrix containing the simulated covariate data (assumed to be marginally uniform).
#' @param marginal_covariate_ranks A dataframe or matrix containing the simulated marginal covariate ranks.
#' @param vine_cor_params A numeric vector specifying the vine correlation parameters.
#' @param topoOrder Optional numeric vector specifying the topological order of the vine structure.
#'        If not provided, the default order \code{1:(n_covariates+1)} is used.
#'
#' @return A list containing:
#' \item{gaussianCopulaFit}{The fitted Gaussian copula model (see \code{fitMVGaussianCopula}).}
#' \item{fullCorrelationMatrix}{The full correlation matrix (including the outcome) computed using \code{computeFullCorMatrix}.}
#' \item{outcomeRankSamples}{A vector of outcome rank samples generated using \code{multivariate_conditional_mean_and_samples}.}
#'
#' @examples
#' \dontrun{
#'   # Assume cov_data and cond_ranks are defined and vine_params is a numeric vector
#'   results <- simulateOutcomeSamples(covariate_data = cov_data, 
#'                                     marginal_covariate_ranks = ranks,
#'                                     vine_cor_params = vine_params)
#'   head(results$outcomeRankSamples)
#' }
simulateMarginalOutcomeSamples <- function(covariate_data,
                                   marginal_covariate_ranks,
                                   vine_cor_params,
                                   topoOrder = NULL,
                                   gaussian_bn_fit = NULL) {
  cov_data_matrix <- as.matrix(covariate_data)
  n_cov <- ncol(cov_data_matrix)
  if (is.null(topoOrder)) {
    topoOrder <- 1:(n_cov + 1)
  }

  # Determine correlation matrix: DAG-constrained or unconstrained
  gaussianCopulaFit <- NULL
  if (!is.null(gaussian_bn_fit)) {
    corMatrixMN <- gaussian_bn_fit$R
  } else {
    gaussianCopulaFit <- fitMVGaussianCopula(dataQuantiles = cov_data_matrix, method = 'itau')
    corMatrixMN <- gaussianCopulaFit$correlationMatrix
  }

  fullCorrelationMatrix <- computeFullCorMatrix(topoOrder, corMatrixMN, vine_cor_params)

  X2_samples <- qnorm(as.matrix(marginal_covariate_ranks))
  outcome_model <- multivariate_conditional_mean_and_samples(X2_samples = X2_samples,
                                                             R = fullCorrelationMatrix)
  outcomeRankSamples <- pnorm(outcome_model$generated_samples)

  return(list(
    gaussianCopulaFit = gaussianCopulaFit,
    gaussian_bn_fit = gaussian_bn_fit,
    fullCorrelationMatrix = fullCorrelationMatrix,
    outcomeRankSamples = outcomeRankSamples
  ))
}

#' Generate Outcome Rank Samples from Vine Copula Simulation with Marginal Ranks
#'
#' This function fits a Gaussian copula to the covariate data, constructs a full correlation
#' matrix for the covariates and outcome (using a provided vine correlation parameter and topological order),
#' and then generates outcome rank samples by conditioning on the simulated covariate ranks.
#'
#' @param covariate_data A dataframe or matrix containing the simulated covariate data (assumed to be marginally uniform).
#' @param cond_covariate_ranks A dataframe or matrix containing the simulated conditional covariate ranks.
#' @param vine_cor_params A numeric vector specifying the vine correlation parameters.
#' @param topoOrder Optional numeric vector specifying the topological order of the vine structure.
#'        If not provided, the default order \code{1:(n_covariates+1)} is used.
#'
#' @return A list containing:
#' \item{gaussianCopulaFit}{The fitted Gaussian copula model (see \code{fitMVGaussianCopula}).}
#' \item{fullCorrelationMatrix}{The full correlation matrix (including the outcome) computed using \code{computeFullCorMatrix}.}
#' \item{outcomeRankSamples}{A vector of outcome rank samples generated using \code{multivariate_conditional_mean_and_samples}.}
#'
#' @examples
#' \dontrun{
#'   # Assume cov_data and cond_ranks are defined and vine_params is a numeric vector
#'   results <- simulateOutcomeSamples(covariate_data = cov_data, 
#'                                     marginal_covariate_ranks = ranks,
#'                                     vine_cor_params = vine_params)
#'   head(results$outcomeRankSamples)
#' }
simulateConditionalOutcomeSamples <- function(covariate_data,
                                              cond_covariate_ranks,
                                           vine_cor_params,
                                           topoOrder = NULL,
                                           gaussian_bn_fit = NULL,
                                           parents = NULL) {
  cov_data_matrix <- as.matrix(covariate_data)
  n_cov <- ncol(cov_data_matrix)
  if (is.null(topoOrder)) {
    topoOrder <- 1:(n_cov + 1)
  }

  # Determine correlation matrix: DAG-constrained (Route B) or unconstrained
  gaussianCopulaFit <- NULL
  if (!is.null(gaussian_bn_fit)) {
    corMatrixMN <- gaussian_bn_fit$R
    if (is.null(parents)) parents <- gaussian_bn_fit$parents
  } else {
    if (!is.null(parents)) {
      gaussian_bn_fit <- fit_reference_gaussian_bn(cov_data_matrix, parents)
      corMatrixMN <- gaussian_bn_fit$R
    } else {
      gaussianCopulaFit <- fitMVGaussianCopula(dataQuantiles = cov_data_matrix, method = 'itau')
      corMatrixMN <- gaussianCopulaFit$correlationMatrix
    }
  }

  fullCorrelationMatrix <- computeFullCorMatrix(topoOrder, corMatrixMN, vine_cor_params)

  # Transform conditional ranks to marginal ranks
  marginal_covariate_ranks <- uncondition_conditional_ranks(
    cond_covariate_ranks, corMatrixMN, parents = parents
  )

  X2_samples <- qnorm(as.matrix(marginal_covariate_ranks))
  outcome_model <- multivariate_conditional_mean_and_samples(X2_samples = X2_samples,
                                                             R = fullCorrelationMatrix)
  outcomeRankSamples <- pnorm(outcome_model$generated_samples)

  return(list(
    gaussianCopulaFit = gaussianCopulaFit,
    gaussian_bn_fit = gaussian_bn_fit,
    fullCorrelationMatrix = fullCorrelationMatrix,
    outcomeRankSamples = outcomeRankSamples
  ))
}


#' Bootstrapped Kendall's Tau Test for Two Vectors
#'
#' This function performs a bootstrapped hypothesis test using Kendall's tau on two input vectors.
#' For each bootstrap iteration, it resamples (with replacement) the data and computes the Kendall's tau test p‑value.
#' Under the null hypothesis of independence, the distribution of these p‑values should be approximately uniform.
#'
#' @param x A numeric vector.
#' @param y A numeric vector.
#' @param n_boot The number of bootstrap iterations (default is 500).
#' @param sample_size The number of samples to draw in each bootstrap iteration (default is the length of \code{x}).
#'
#' @return A numeric vector of bootstrapped p‑values from Kendall's tau tests.
#'
#' @examples
#' \dontrun{
#'   # Assume x and y are numeric vectors of equal length
#'   p_values <- bootstrappedKendallTest(x, y, n_boot = 500)
#'   hist(p_values, main = "Bootstrapped Kendall's Tau p-values", xlab = "p-value")
#' }
bootstrappedKendallTest <- function(x, y, n_boot = 500, sample_size = length(x)) {
  if (length(x) != length(y)) {
    stop("x and y must be of the same length")
  }
  
  n <- length(x)
  p_values <- numeric(n_boot)
  
  # Create a progress bar.
  pb <- txtProgressBar(min = 0, max = n_boot, style = 3)
  
  for (i in 1:n_boot) {
    # Draw a bootstrap sample (with replacement)
    boot_idx <- sample(1:n, sample_size, replace = TRUE)
    boot_x <- x[boot_idx]
    boot_y <- y[boot_idx]
    
    # Compute the Kendall's tau test p-value.
    test_result <- cor.test(boot_x, boot_y, method = "kendall")
    p_values[i] <- test_result$p.value
    
    # Update the progress bar.
    setTxtProgressBar(pb, i)
  }
  
  close(pb)
  return(p_values)
}


#' Bootstrapped Kernel Conditional Independence Test (KCI) with Improved Settings and Progress Bar
#'
#' This function performs a bootstrapped conditional independence test using the KCI test with improved
#' settings. It resamples the data (with replacement) and, for each bootstrap iteration, calls the KCI
#' function with automatically tuned kernel hyperparameters (using GP regression) and Gamma approximation.
#'
#' A progress bar is displayed during the bootstrapping iterations.
#'
#' @param x A numeric vector.
#' @param y A numeric vector.
#' @param z A matrix or dataframe of conditioning variables (one row per observation).
#' @param n_boot The number of bootstrap iterations (default is 500).
#' @param sample_size The number of observations to sample in each bootstrap iteration (default is the length of x).
#'
#' @return A numeric vector of bootstrap p‑values from the KCI tests.
#'
#' @examples
#' \dontrun{
#'   # Suppose X, Y are numeric vectors and Z is a matrix of conditioning variables.
#'   p_values <- bootstrappedKCI(X, Y, Z, n_boot = 500)
#'   hist(p_values, main = "Bootstrapped KCI p-values", xlab = "p-value")
#' }
bootstrappedKCI <- function(x, y, z, n_boot = 500, sample_size = length(x)) {
  # Check that x and y have the same length.
  if (length(x) != length(y)) {
    stop("x and y must be of the same length")
  }
  
  # Ensure that the number of rows in z matches the length of x.
  if (nrow(as.matrix(z)) != length(x)) {
    stop("The number of rows in z must match the length of x and y")
  }
  
  n <- length(x)
  p_values <- numeric(n_boot)
  
  # Create a progress bar.
  pb <- txtProgressBar(min = 0, max = n_boot, style = 3)
  
  for (i in 1:n_boot) {
    # Draw bootstrap sample indices with replacement.
    boot_idx <- sample(1:n, sample_size, replace = TRUE)
    
    boot_x <- x[boot_idx]
    boot_y <- y[boot_idx]
    boot_z <- as.matrix(z)[boot_idx, , drop = FALSE]
    
    # Run the Kernel Conditional Independence test with the specified settings.
    kci_result <- tryCatch({
      KCI(boot_x, boot_y, boot_z,
          width      = 0,
          alpha      = 0.05,
          unbiased   = FALSE,
          gammaApprox= FALSE,
          GP         = TRUE,
          nRepBs     = 1000,
          lambda     = 0.001,
          thresh     = 1e-05,
          numEig     = length(boot_x),
          verbose    = FALSE)
    }, error = function(e) {
      message("Error in KCI on bootstrap iteration ", i, ": ", e$message)
      return(list(pvalue = NA))
    })
    
    # Save the bootstrap p-value.
    p_values[i] <- kci_result$pvalue
    
    # Update the progress bar.
    setTxtProgressBar(pb, i)
  }
  
  # Close the progress bar.
  close(pb)
  
  return(p_values)
}


#' Bootstrapped Conditional Independence Test using bnlearn::ci.test
#'
#' This function performs a bootstrapped conditional independence test using bnlearn's ci.test.
#' For each bootstrap iteration, it resamples the data (with replacement) and computes the p-value
#' from ci.test. Under the null hypothesis of conditional independence, the distribution of p-values
#' should be approximately uniform.
#'
#' @param x A numeric vector.
#' @param y A numeric vector.
#' @param z A matrix or dataframe of conditioning variables (one row per observation).
#' @param n_boot The number of bootstrap iterations (default is 500).
#' @param sample_size The number of observations to sample in each bootstrap iteration (default is length(x)).
#' @param test The conditional independence test to use (default is "cor" for linear Gaussian tests).
#'
#' @return A numeric vector of bootstrap p-values from ci.test.
#'
#' @examples
#' \dontrun{
#'   # Generate some example data:
#'   set.seed(123)
#'   n <- 1000
#'   x <- rnorm(n)
#'   y <- rnorm(n)
#'   z <- data.frame(Z1 = rnorm(n), Z2 = runif(n))
#'   
#'   # Run the bootstrapped CI test:
#'   p_values <- bootstrappedCITest(x, y, z, n_boot = 500, test = "cor")
#'   hist(p_values, main = "Bootstrapped ci.test p-values", xlab = "p-value")
#' }
bootstrappedCITest <- function(x, y, z, n_boot = 500, sample_size = length(x), test = "cor") {
  # Check that x and y have the same length.
  if (length(x) != length(y)) {
    stop("x and y must be of the same length")
  }
  
  # Ensure that the number of rows in z matches the length of x.
  if (nrow(as.matrix(z)) != length(x)) {
    stop("The number of rows in z must match the length of x and y")
  }
  
  n <- length(x)
  p_values <- numeric(n_boot)
  
  # Create a progress bar.
  pb <- txtProgressBar(min = 0, max = n_boot, style = 3)
  
  for (i in 1:n_boot) {
    # Draw bootstrap sample indices with replacement.
    boot_idx <- sample(1:n, sample_size, replace = TRUE)
    
    # Resample the data.
    boot_x <- x[boot_idx]
    boot_y <- y[boot_idx]
    boot_z <- as.data.frame(as.matrix(z)[boot_idx, , drop = FALSE])
    
    # Build a data frame with standardized column names.
    df <- data.frame(A = boot_x, B = boot_y, boot_z)
    # The conditioning variables are the remaining columns.
    cond_vars <- names(boot_z)
    
    # Run bnlearn's conditional independence test.
    test_result <- tryCatch({
      ci.test(x = "A", y = "B", z = cond_vars, data = df, test = test)
    }, error = function(e) {
      message("Error in ci.test on iteration ", i, ": ", e$message)
      return(list(p.value = NA))
    })
    
    # Extract the p-value (if missing, assign NA).
    p_val <- test_result$p.value
    if (is.null(p_val) || length(p_val) == 0)
      p_val <- NA
    p_values[i] <- p_val
    
    # Update the progress bar.
    setTxtProgressBar(pb, i)
  }
  
  # Close the progress bar.
  close(pb)
  
  return(p_values)
}

#' Bootstrapped Conditional Independence Test using gcm.test
#'
#' This function performs a bootstrapped conditional independence test using
#' gcm.test() from the GeneralisedCovarianceMeasure package. In each bootstrap iteration,
#' it resamples the data (with replacement) and calls gcm.test() with the supplied arguments.
#' The p‑values are collected and returned.
#'
#' @param x A numeric vector.
#' @param y A numeric vector.
#' @param z A matrix or dataframe of conditioning variables (with one row per observation).
#' @param n_boot The number of bootstrap iterations (default is 500).
#' @param sample_size The number of observations to sample in each bootstrap iteration (default is the length of x).
#' @param verbose A logical flag; if TRUE, messages will be printed for errors.
#' @param ... Additional arguments passed to gcm.test().
#'
#' @return A numeric vector of bootstrap p‑values from gcm.test.
#'
#' @examples
#' \dontrun{
#'   library(GeneralisedCovarianceMeasure)
#'   set.seed(1)
#'   n <- 100
#'   x <- rnorm(n)
#'   y <- rnorm(n)
#'   z <- data.frame(Z1 = rnorm(n), Z2 = runif(n))
#'   p_vals <- bootstrappedCondIndTest_GCM(x, y, z, n_boot = 500, sample_size = n)
#'   hist(p_vals, main = "Bootstrapped gcm.test p-values", xlab = "p-value")
#' }
bootstrappedCondIndTest_GCM <- function(x, y, z, 
                                        n_boot = 500, 
                                        sample_size = length(x), 
                                        verbose = FALSE, 
                                        ...) {
  # Check that x and y have the same length.
  if (length(x) != length(y)) {
    stop("x and y must be of the same length")
  }
  
  # Ensure that the number of rows in z matches the length of x.
  if (nrow(as.matrix(z)) != length(x)) {
    stop("The number of rows in z must match the length of x and y")
  }
  
  n <- length(x)
  p_values <- numeric(n_boot)
  
  # Create a progress bar.
  pb <- txtProgressBar(min = 0, max = n_boot, style = 3)
  
  for (i in 1:n_boot) {
    # Draw bootstrap sample indices with replacement.
    boot_idx <- sample(1:n, sample_size, replace = TRUE)
    
    boot_x <- x[boot_idx]
    boot_y <- y[boot_idx]
    boot_z <- as.data.frame(as.matrix(z)[boot_idx, , drop = FALSE])
    
    # Call gcm.test() with the supplied extra parameters.
    result <- tryCatch({
      gcm.test(boot_x, boot_y, boot_z, ...)
    }, error = function(e) {
      if (verbose) message("Error in gcm.test on iteration ", i, ": ", e$message)
      return(list(p.value = NA))
    })
    
    p_val <- result$p.value
    if (is.null(p_val) || length(p_val) == 0) p_val <- NA
    p_values[i] <- p_val
    
    setTxtProgressBar(pb, i)
  }
  
  close(pb)
  return(p_values)
}

# Make sure you have CondIndTests installed:
# install.packages("CondIndTests")
library(CondIndTests)

#' Bootstrapped Conditional Independence Test using CondIndTest
#'
#' This function performs a bootstrapped conditional independence test using
#' CondIndTest from the CondIndTests package. It resamples the data (with replacement)
#' and, for each bootstrap iteration, calls CondIndTest with the specified settings.
#' The p‑values are collected and returned. The function uses a progress bar to show progress.
#'
#' @param x A numeric vector.
#' @param y A numeric vector.
#' @param z A matrix or dataframe of conditioning variables (one row per observation).
#' @param n_boot The number of bootstrap iterations (default is 500).
#' @param sample_size The number of observations to sample in each bootstrap iteration (default is length(x)).
#' @param method The method to use in CondIndTest (default is "KCI").
#' @param alpha Significance level for the test (default is 0.05).
#' @param parsMethod A list of parameters for the kernel hyperparameter selection in CondIndTest (default is an empty list).
#' @param verbose A logical flag passed to CondIndTest (default is FALSE).
#'
#' @return A numeric vector of bootstrap p‑values from CondIndTest.
#'
#' @examples
#' \dontrun{
#'   set.seed(1)
#'   n <- 100
#'   Z <- rnorm(n)
#'   X <- 4 + 2 * Z + rnorm(n)
#'   Y <- 3 * X^2 + Z + rnorm(n)
#'   # In these data X and Y are NOT conditionally independent given Z.
#'   p_vals <- bootstrappedCondIndTest_CIT(X, Y, Z, n_boot = 500, sample_size = n)
#'   cat("Bootstrapped p-values:\n")
#'   print(p_vals)
#' }
bootstrappedCondIndTest_CIT <- function(x, y, z, 
                                        n_boot = 500, 
                                        sample_size = length(x),
                                        method = "KCI", 
                                        alpha = 0.05, 
                                        parsMethod = list(), 
                                        verbose = FALSE) {
  # Check that x and y have the same length.
  if (length(x) != length(y)) {
    stop("x and y must be of the same length")
  }
  # Ensure that the number of rows in z matches the length of x.
  if (nrow(as.matrix(z)) != length(x)) {
    stop("The number of rows in z must match the length of x and y")
  }
  
  n <- length(x)
  p_values <- numeric(n_boot)
  
  # Create a progress bar.
  pb <- txtProgressBar(min = 0, max = n_boot, style = 3)
  
  for (i in 1:n_boot) {
    # Draw bootstrap sample indices with replacement.
    boot_idx <- sample(1:n, sample_size, replace = TRUE)
    boot_x <- x[boot_idx]
    boot_y <- y[boot_idx]
    boot_z <- as.data.frame(as.matrix(z)[boot_idx, , drop = FALSE])
    
    # Call CondIndTest with the specified settings.
    result <- tryCatch({
      CondIndTest(boot_x, boot_y, boot_z, 
                  method = method, 
                  alpha = alpha, 
                  parsMethod = parsMethod, 
                  verbose = verbose)
    }, error = function(e) {
      message("Error in CondIndTest on iteration ", i, ": ", e$message)
      return(list(pvalue = NA))
    })
    
    # Extract the p-value (if missing, assign NA)
    p_val <- result$pvalue
    if (is.null(p_val) || length(p_val) == 0) {
      p_val <- NA
    }
    p_values[i] <- p_val
    
    # Update progress bar.
    setTxtProgressBar(pb, i)
  }
  
  close(pb)
  return(p_values)
}
