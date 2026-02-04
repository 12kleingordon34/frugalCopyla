#' @title Simulation Functions for R-Vine Copula Models
#' @description Functions for simulating data from R-vine copula models and
#'   reparameterizing vines for nonparanormal approximation.
#' @name simulate
NULL

#' Simulate data from an R-vine copula model
#'
#' This function generates synthetic data from an R-vine copula model specified
#' by the given structure, family, and parameter matrices.
#'
#' @param structureMatrix A matrix specifying the structure of the R-vine copula model.
#'   The structure matrix encodes the dependence structure among variables.
#' @param familyMatrix A matrix specifying the families of bivariate copulas used
#'   in the R-vine model. Each entry corresponds to a pair of variables.
#' @param parameterMatrix A matrix specifying the parameters of the bivariate copulas
#'   used in the R-vine model.
#' @param sampleSize Integer. The number of observations to simulate. Default is 300.
#' @param seed Integer. Random seed for reproducibility. Default is 123.
#'
#' @return A list containing:
#' \describe{
#'   \item{RVM}{The RVineMatrix object defining the vine structure}
#'   \item{simdata}{A matrix containing the simulated data with rows representing
#'     observations and columns representing variables}
#' }
#'
#' @examples
#' \dontrun{
#' # Simple 3-variable example
#' structureMatrix <- matrix(c(3, 0, 0, 2, 3, 0, 1, 2, 3), ncol = 3)
#' familyMatrix <- matrix(c(0, 0, 0, 0, 1, 0, 0, 1, 0), ncol = 3)
#' parameterMatrix <- matrix(c(0, 0, 0, 0, 0.5, 0, 0, 0.3, 0), ncol = 3)
#' result <- simulateRVineData(structureMatrix, familyMatrix, parameterMatrix, 100)
#' }
#'
#' @importFrom VineCopula RVineMatrix RVineSim
#' @export
simulateRVineData <- function(structureMatrix, familyMatrix, parameterMatrix,
                               sampleSize = 300, seed = 123) {
  # Define variable names dynamically based on the structure matrix's dimension
  varNames <- paste0("V", 1:(max(structureMatrix[structureMatrix > 0])))

  # Define the RVineMatrix object
  RVM <- VineCopula::RVineMatrix(
    Matrix = structureMatrix,
    family = familyMatrix,
    par = parameterMatrix,
    names = varNames
  )

  # Set seed for reproducibility
  set.seed(seed)

  # Simulate data from the defined R-vine model
  simdata <- VineCopula::RVineSim(sampleSize, RVM)

  return(list(RVM = RVM, simdata = simdata))
}


#' Simulate and Reparameterize Vine Copula Data
#'
#' Simulates data from a specified vine copula model, fits a multivariate Gaussian
#' copula to the simulated data, reparameterizes the vine copula according to the
#' nonparanormal approximation using partial correlations, and simulates new data
#' from the reparameterized vine.
#'
#' This process aligns the copula model closer to the empirical correlations
#' observed in the data, especially for complex dependencies.
#'
#' @param structureMatrix Square, lower triangular matrix specifying the R-vine structure.
#' @param familyMatrix Square, lower triangular matrix specifying the family of the vine copulae.
#' @param parameterMatrix Square, lower triangular matrix specifying the parameters of the vine copulae.
#' @param sampleSize Integer. The number of samples to draw from the vine.
#' @param topoOrder Numeric vector. The topological order of the vine copula.
#' @param vineCorParams Numeric vector. The correlation parameters between pretreatment
#'   confounders and Y.
#' @param seed Integer. Random seed for reproducibility. Default is 1.
#'
#' @return A list containing:
#' \describe{
#'   \item{oldVineOutput}{The simulated data from the original vine specification}
#'   \item{newVineOutput}{The simulated data from the reparameterized vine}
#'   \item{standardized_precision_matrix_np}{The standardized precision matrix of the new simulated data}
#'   \item{fullCorMatrixMN}{The full correlation matrix including the outcome}
#' }
#'
#' @examples
#' \dontrun{
#' D <- 6
#' sampleSize <- 5000
#' topoOrder <- 1:D
#' vineCorParams <- c(0.5)
#' results <- simulateAndReparameterizeVine(structureMatrix, familyMatrix,
#'                                          parameterMatrix, sampleSize,
#'                                          topoOrder, vineCorParams)
#' }
#'
#' @export
simulateAndReparameterizeVine <- function(structureMatrix, familyMatrix, parameterMatrix,
                                           sampleSize, topoOrder, vineCorParams, seed = 1) {
  D <- nrow(structureMatrix)

  # Initial vine copula simulation
  oldVineOutput <- simulateRVineData(structureMatrix, familyMatrix, parameterMatrix,
                                      sampleSize, seed)

  # Fit a multivariate Gaussian copula to the simulated data
  mvgFit <- fitMVGaussianCopula(oldVineOutput$simdata, method = 'itau')

  # Reparameterization settings
  corMatrixMN <- mvgFit$correlationMatrix[1:(D - 1), 1:(D - 1)]
  fullCorMatrixMN <- computeFullCorMatrix(topoOrder, corMatrixMN, vineCorParams)
  partialCors <- computePartialCorrelations(fullCorMatrixMN, topoOrder)

  # Update vine matrices based on the nonparanormal approximation
  updatedVine <- updateVineMatrices(structureMatrix, familyMatrix, parameterMatrix, partialCors)

  # Simulate data from the updated vine
  newVineOutput <- simulateRVineData(
    updatedVine$StructureMatrix,
    updatedVine$FamilyMatrix,
    updatedVine$ParameterMatrix,
    sampleSize,
    seed = seed
  )

  # Compute the standardized precision matrix for the new simulated data
  standardized_precision_matrix_np <- compute_standardized_precision_matrix(newVineOutput$simdata)

  return(list(
    oldVineOutput = oldVineOutput,
    newVineOutput = newVineOutput,
    standardized_precision_matrix_np = standardized_precision_matrix_np,
    fullCorMatrixMN = fullCorMatrixMN
  ))
}
