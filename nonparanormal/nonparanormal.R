library(bnlearn)
library(CondIndTests)
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
computeFullCorMatrix <- function(invTopoOrder, corMatrixMN, vineCorParams) {
  # Ensure the input correlation matrix matches the topological order size
  D <- length(invTopoOrder)
  # N <- length(vineCorParams)
  # M <- D - N
  if (!all(dim(corMatrixMN) == c(D, D))) {
    stop("Dimension mismatch: corMatrixMN should match the length of invTopoOrder")
  }
  
  # Initialize full correlation matrix with diagonal ones
  fullCorMatrix <- diag(1, D+1)
  fullCorMatrix[1:D, 1:D] <- corMatrixMN
  
  # Add the M+Nth variable (last of N) and its marginal correlation with Y
  fullCorMatrix[D, D + 1] <- fullCorMatrix[D + 1, D] <- head(vineCorParams, 1)
  
  # Define the initial conditioning set B as the last variable
  conditioningSet <- invTopoOrder[1]
  
  # Loop over N variables in reverse topological order
  # for (i in (D-1):1) {
  vineCorParamsReduced <- vineCorParams[2:D]
  for (i in seq_along(invTopoOrder[2:D])) {  
    rho <- vineCorParamsReduced[i]
    # if (i %in% invTopoOrder[(M+1):D]) {
    #   # Set A is Y and the current variable
    #   rho <- vineCorParams[which(invTopoOrder[(D-N+1):D] == i)]
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
#' @param cond_ranks A matrix (or dataframe) of conditional ranks. The first column is assumed to be
#'        unconditional (U_{Z1}), the second column is U_{Z2|Z1}, the third is U_{Z3|Z1,Z2}, etc.
#' @param R A correlation matrix corresponding to the full Gaussian copula that models the joint distribution.
#'
#' @return A matrix of the same dimensions as \code{cond_ranks} containing the "unconditioned" values
#'         on the Gaussian (normal quantile) scale.
#'
#' @examples
#' # Suppose we have a 3-variable example:
#' cond_ranks <- matrix(c(runif(5), runif(5), runif(5)), ncol = 3)
#' R <- matrix(c(1, 0.5, 0.3,
#'               0.5, 1, 0.4,
#'               0.3, 0.4, 1), nrow = 3, byrow = TRUE)
#' marginal_values <- uncondition_conditional_ranks(cond_ranks, R)
uncondition_conditional_ranks <- function(cond_ranks, R) {
  # Ensure cond_ranks is a matrix
  cond_ranks <- as.matrix(cond_ranks)
  n <- nrow(cond_ranks)
  d <- ncol(cond_ranks)
  
  # Prepare a matrix to store the marginal (unconditioned) values.
  marginal_values <- matrix(NA, n, d)
  
  # Process each observation (each row) individually.
  for (i in 1:n) {
    x <- numeric(d)
    # The first variable is unconditional.
    x[1] <- qnorm(cond_ranks[i, 1])
    
    # For subsequent variables, uncondition using the Gaussian translation.
    if (d > 1) {
      for (j in 2:d) {
        # Coerce r_vec into a 1-row matrix.
        r_vec <- matrix(R[j, 1:(j-1)], nrow = 1)
        # Ensure the submatrix is a matrix.
        R_sub <- R[1:(j-1), 1:(j-1), drop = FALSE]
        # Coerce the previously computed x's into a column vector.
        x_prev <- matrix(x[1:(j-1)], ncol = 1)
        
        # Compute the conditional mean.
        mu_j <- as.numeric(r_vec %*% solve(R_sub) %*% x_prev)
        # Compute the conditional variance.
        sigma2_j <- 1 - as.numeric(r_vec %*% solve(R_sub) %*% t(r_vec))
        sigma_j <- sqrt(sigma2_j)
        
        # "Uncondition" the j-th variable by converting its conditional rank into a marginal Gaussian value.
        x[j] <- mu_j + sigma_j * qnorm(cond_ranks[i, j])
      }
    }
    marginal_values[i, ] <- x
  }
  
  return(pnorm(marginal_values))
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
                                   topoOrder = NULL) {
  ## Step 1: Fit the Gaussian copula to the covariate data.
  cov_data_matrix <- as.matrix(covariate_data)
  gaussianCopulaFit <- fitMVGaussianCopula(dataQuantiles = cov_data_matrix, method = 'itau')
  
  ## Step 2: Construct the full correlation matrix.
  n_cov <- ncol(cov_data_matrix)
  if (is.null(topoOrder)) {
    topoOrder <- 1:(n_cov + 1)
  }
  # Use the correlation matrix from the Gaussian copula fit.
  corMatrixMN <- gaussianCopulaFit$correlationMatrix
  fullCorrelationMatrix <- computeFullCorMatrix(topoOrder, corMatrixMN, vine_cor_params)
  
  ## Step 3: Generate outcome rank samples.
  # Transform the simulated conditional covariate ranks with qnorm (to obtain normal scores).
  X2_samples <- qnorm(as.matrix(marginal_covariate_ranks))
  outcome_model <- multivariate_conditional_mean_and_samples(X2_samples = X2_samples, 
                                                             R = fullCorrelationMatrix)
  outcomeRankSamples <- pnorm(outcome_model$generated_samples)
  
  return(list(
    gaussianCopulaFit = gaussianCopulaFit,
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
                                           topoOrder = NULL) {
  ## Step 1: Fit the Gaussian copula to the covariate data.
  cov_data_matrix <- as.matrix(covariate_data)
  gaussianCopulaFit <- fitMVGaussianCopula(dataQuantiles = cov_data_matrix, method = 'itau')
  
  ## Step 2: Construct the full correlation matrix.
  n_cov <- ncol(cov_data_matrix)
  if (is.null(topoOrder)) {
    topoOrder <- 1:(n_cov + 1)
  }
  # Use the correlation matrix from the Gaussian copula fit.
  corMatrixMN <- gaussianCopulaFit$correlationMatrix
  fullCorrelationMatrix <- computeFullCorMatrix(topoOrder, corMatrixMN, vine_cor_params)
  
  
  marginal_covariate_ranks <- uncondition_conditional_ranks(cond_covariate_ranks, corMatrixMN)
  ## Step 3: Generate outcome rank samples.
  # Transform the simulated conditional covariate ranks with qnorm (to obtain normal scores).
  X2_samples <- qnorm(as.matrix(marginal_covariate_ranks))
  outcome_model <- multivariate_conditional_mean_and_samples(X2_samples = X2_samples, 
                                                             R = fullCorrelationMatrix)
  outcomeRankSamples <- pnorm(outcome_model$generated_samples)
  
  return(list(
    gaussianCopulaFit = gaussianCopulaFit,
    fullCorrelationMatrix = fullCorrelationMatrix,
    outcomeRankSamples = outcomeRankSamples
  ))
}

#' Bootstrapped Kendall's Tau Test for Two Vectors
#'
#' This function performs a bootstrapped hypothesis test using Kendall's tau on two input vectors.
#' In each bootstrap iteration the function resamples (with replacement) the data and computes the Kendall
#' correlation test p‑value. Under independence, the distribution of these p‑values should be approximately uniform.
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
  
  for (i in 1:n_boot) {
    # Draw a bootstrap sample (with replacement)
    boot_idx <- sample(1:n, sample_size, replace = TRUE)
    boot_x <- x[boot_idx]
    boot_y <- y[boot_idx]
    
    # Compute the Kendall's tau test p-value.
    test_result <- cor.test(boot_x, boot_y, method = "kendall")
    p_values[i] <- test_result$p.value
  }
  
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