#' @title Vine Copula Transformation Functions
#' @description Functions for reparameterizing vine copula matrices for
#'   nonparanormal approximation.
#' @name vine_transform
NULL

#' Reparameterize Vine Matrices for Non-Paranormal Approximation
#'
#' Updates vine copula matrices (structureMatrix, familyMatrix, parameterMatrix)
#' according to transformation rules for nonparanormal approximation. This allows
#' for more flexible modeling of dependencies by applying transformations that
#' align with Gaussian copula structures.
#'
#' @param structureMatrix A square, lower triangular matrix representing the R-vine
#'   structure to be adjusted.
#' @param familyMatrix A square, lower triangular matrix representing the family
#'   of the vine copulae, reflecting the types of dependencies among variables.
#' @param parameterMatrix A square, lower triangular matrix representing the
#'   parameters of the vine copulae.
#' @param partialCors A numeric vector of size D-1, where D is the dimension of
#'   the matrices, representing partial correlations for the nonparanormal
#'   approximation.
#'
#' @return A list containing:
#' \describe{
#'   \item{StructureMatrix}{Updated structure matrix}
#'   \item{FamilyMatrix}{Updated family matrix}
#'   \item{ParameterMatrix}{Updated parameter matrix with partial correlations}
#' }
#'
#' @examples
#' \dontrun{
#' D <- 4
#' structureMatrix <- matrix(c(4, 0, 0, 0, 3, 4, 0, 0, 2, 3, 4, 0, 1, 2, 3, 4),
#'                           nrow = 4, byrow = TRUE)
#' familyMatrix <- matrix(0, 4, 4)
#' parameterMatrix <- matrix(0, 4, 4)
#' partialCors <- c(0.1, 0.2, 0.3)
#'
#' results <- updateVineMatrices(structureMatrix, familyMatrix,
#'                               parameterMatrix, partialCors)
#' }
#'
#' @export
updateVineMatrices <- function(structureMatrix, familyMatrix, parameterMatrix, partialCors) {
  D <- nrow(structureMatrix)

  # Validate input dimensions and types
  if (D != ncol(structureMatrix) ||
      D != nrow(familyMatrix) ||
      D != ncol(familyMatrix) ||
      D != nrow(parameterMatrix) ||
      D != ncol(parameterMatrix)) {
    stop("All input matrices must be square and of the same dimension.")
  }

  if (length(partialCors) != D - 1) {
    stop("partialCors must have length D-1, where D is the dimension of the matrices.")
  }

  # Update structureMatrix
  newStructureMatrix <- (structureMatrix - 1)
  newStructureMatrix[1, 1] <- newStructureMatrix[1, 1] + D
  newStructureMatrix[upper.tri(newStructureMatrix)] <- 0

  # Update familyMatrix
  newFamilyMatrix <- familyMatrix
  newFamilyMatrix[, 2:(D - 1)] <- familyMatrix[, 1:(D - 2)]
  newFamilyMatrix[(2:D), 1] <- 1

  # Update parameterMatrix
  newParameterMatrix <- parameterMatrix
  newParameterMatrix[D, 2:(D - 1)] <- parameterMatrix[D, 1:(D - 2)]
  newParameterMatrix[(D:2), 1] <- partialCors

  return(list(
    StructureMatrix = newStructureMatrix,
    FamilyMatrix = newFamilyMatrix,
    ParameterMatrix = newParameterMatrix
  ))
}
