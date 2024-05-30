library(copula)
library(ppcor)
library(VineCopula)

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
simulateRVineData <- function(structureMatrix, familyMatrix, parameterMatrix, sampleSize = 300) {
  # Ensure matrices are in the correct format
  # structureMatrix <- matrix(structureMatrix, ncol = sqrt(length(structureMatrix)))
  # familyMatrix <- matrix(familyMatrix, ncol = sqrt(length(familyMatrix)))
  # parameterMatrix <- matrix(parameterMatrix, ncol = sqrt(length(parameterMatrix)))
  
  # Define variable names dynamically based on the structure matrix's dimension
  varNames <- paste0("V", 1:(max(structureMatrix[structureMatrix > 0])))
  
  # Define the RVineMatrix object
  RVM <- RVineMatrix(Matrix = structureMatrix, family = familyMatrix, par = parameterMatrix, names = varNames)
  
  # Set seed for reproducibility (optional)
  set.seed(123)
  
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
  MplusN <- length(topoOrder) - 1
  N <- length(vineCorParams)
  M <- MplusN - N
  if (!all(dim(corMatrixMN) == c(MplusN, MplusN))) {
    stop("Dimension mismatch: corMatrixMN should match the length of topoOrder")
  }
  
  # Initialize full correlation matrix with diagonal ones
  fullCorMatrix <- diag(1, MplusN+1)
  fullCorMatrix[1:MplusN, 1:MplusN] <- corMatrixMN
  
  # Add the M+Nth variable (last of N) and its marginal correlation with Y
  fullCorMatrix[MplusN, MplusN + 1] <- fullCorMatrix[MplusN + 1, MplusN] <- tail(vineCorParams, 1)
  
  # Define the initial conditioning set B as the last variable
  conditioningSet <- MplusN
  
  # Loop over N variables in reverse topological order
  for (i in (MplusN-1):1) {
    if (i %in% topoOrder[(M+1):MplusN]) {
      # Set A is Y and the current variable
      rho <- vineCorParams[which(topoOrder[(MplusN-N+1):MplusN] == i)]
    } else {
      rho <- 0 
    }
    # Set A is Y and the current variable
    Sigma_AB <- matrix(fullCorMatrix[c(i, MplusN+1), c(conditioningSet)], nrow=2)
    Sigma_BB <- corMatrixMN[conditioningSet, conditioningSet]

    # Compute marginal correlation
    rho_marginal <- computeConditionalCovariance(rho, Sigma_AB, Sigma_BB)
    
    # Append the computed marginal correlation to the full correlation matrix
    fullCorMatrix[i, MplusN + 1] <- fullCorMatrix[MplusN + 1, i] <- rho_marginal
    
    
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
simulateAndReparameterizeVine <- function(structureMatrix, familyMatrix, parameterMatrix, sampleSize, topoOrder, vineCorParams) {
  library(copula)
  library(VineCopula)
  
  # Initial vine copula simulation
  oldVineOutput <- simulateRVineData(structureMatrix, familyMatrix, parameterMatrix, sampleSize)
  
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
    sampleSize
  )
  
  # Compute the standardized precision matrix for the new simulated data
  standardized_precision_matrix_np <- compute_standardized_precision_matrix(newVineOutput$simdata)
  
  return(list(oldVineOutput = oldVineOutput, newVineOutput = newVineOutput, standardized_precision_matrix_np = standardized_precision_matrix_np))
}

simulateAndPlot <- function(structureMatrix, familyMatrix, sampleSize, topoOrder, normal_corr_values, general_dep, general_family, seed=1) {
  plot_list <- list() # Initialize an empty list to store ggplot objects
  set.seed(seed)
  for (normal_corr in normal_corr_values) {
    # Update the normal correlation parameter in the parameter matrix
    parameterMatrix <- matrix(
      c(0, 0, 0, 0, 0, general_dep, 
        0, 0, 0, 0, 0, general_dep,
        0, 0, 0, 0, 0, general_dep,
        0, 0, 0, 0, 0, general_dep,
        0, 0, 0, 0, 0, normal_corr,
        0, 0, 0, 0, 0, 0), ncol=D)
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
    newVineOutput <- npReparamVine$newVineOutput
    
    margins <- oldVineOutput$simdata
    margins_np <- newVineOutput$simdata
    
    # Calculate H-functions for both old and new vine outputs
    F4_5_np <- BiCopHfunc(margins_np[,4], margins_np[,5], family=general_family, par=general_dep)$hfunc2
    F6_5_np <- BiCopHfunc(margins_np[,6], margins_np[,5], family=1, par=normal_corr)$hfunc2
    
    F4_5 <- BiCopHfunc(margins[,4], margins[,5], family=general_family, par=general_dep)$hfunc2
    F6_5 <- BiCopHfunc(margins[,6], margins[,5], family=1, par=normal_corr)$hfunc2
    
    # Generate plots
    p1 <- ggplot(data.frame(F4_5 = F4_5_np, F6_5 = F6_5_np), aes(x = F4_5, y = F6_5)) +
      stat_density_2d(aes(fill = ..level..), geom = "polygon") +  # Use stat_density_2d for filled contours
      scale_fill_viridis_c() +  # Adds a color gradient based on density levels
      labs(x = "F4_5", y = "F6_5", title = paste("NP Contour Plot of F4_5 vs F6_5 (corr =", normal_corr, ")")) +
      theme_minimal()
    
    
    p2 <- ggplot(data.frame(F4_5 = F4_5, F6_5 = F6_5), aes(x = F4_5, y = F6_5)) +
      stat_density_2d(aes(fill = ..level..), geom = "polygon") +  # Use stat_density_2d for filled contours
      scale_fill_viridis_c() +  # Adds a color gradient based on density levels
      labs(x = "F4_5", y = "F6_5", title = paste("True Contour Plot of F4_5 vs F6_5 (corr =", normal_corr, ")")) +
      theme_minimal()
    
    # Add plots to the list
    plot_list[[length(plot_list) + 1]] <- p1
    plot_list[[length(plot_list) + 1]] <- p2
  }
  
  # Combine all plots into a grid
  do.call(grid.arrange, c(plot_list, ncol = 2))
}

