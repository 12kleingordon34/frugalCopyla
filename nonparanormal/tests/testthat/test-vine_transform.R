# Test suite for vine transformation functions

test_that("updateVineMatrices returns correct dimensions", {
  D <- 4
  structureMatrix <- matrix(c(
    4, 0, 0, 0,
    3, 4, 0, 0,
    2, 3, 4, 0,
    1, 2, 3, 4
  ), nrow = 4, byrow = TRUE)
  familyMatrix <- matrix(0, D, D)
  familyMatrix[D, 1:(D-1)] <- 1
  parameterMatrix <- matrix(0, D, D)
  parameterMatrix[D, 1:(D-1)] <- 0.5
  partialCors <- c(0.1, 0.2, 0.3)

  result <- updateVineMatrices(structureMatrix, familyMatrix, parameterMatrix, partialCors)

  expect_equal(dim(result$StructureMatrix), c(D, D))
  expect_equal(dim(result$FamilyMatrix), c(D, D))
  expect_equal(dim(result$ParameterMatrix), c(D, D))
})

test_that("updateVineMatrices preserves structure validity", {
  D <- 4
  structureMatrix <- matrix(c(
    4, 0, 0, 0,
    3, 4, 0, 0,
    2, 3, 4, 0,
    1, 2, 3, 4
  ), nrow = 4, byrow = TRUE)
  familyMatrix <- matrix(0, D, D)
  familyMatrix[D, 1:(D-1)] <- 1
  parameterMatrix <- matrix(0, D, D)
  parameterMatrix[D, 1:(D-1)] <- 0.5
  partialCors <- c(0.1, 0.2, 0.3)

  result <- updateVineMatrices(structureMatrix, familyMatrix, parameterMatrix, partialCors)

  # Upper triangle should be zero
  expect_true(all(result$StructureMatrix[upper.tri(result$StructureMatrix)] == 0))
})

test_that("updateVineMatrices throws error for dimension mismatch", {
  D <- 4
  structureMatrix <- matrix(1, D, D)
  familyMatrix <- matrix(1, D - 1, D - 1)  # Wrong dimension
  parameterMatrix <- matrix(0, D, D)
  partialCors <- rep(0.5, D - 1)

  expect_error(
    updateVineMatrices(structureMatrix, familyMatrix, parameterMatrix, partialCors),
    "same dimension"
  )
})

test_that("updateVineMatrices throws error for wrong partialCors length", {
  D <- 4
  structureMatrix <- matrix(c(
    4, 0, 0, 0,
    3, 4, 0, 0,
    2, 3, 4, 0,
    1, 2, 3, 4
  ), nrow = 4, byrow = TRUE)
  familyMatrix <- matrix(0, D, D)
  parameterMatrix <- matrix(0, D, D)
  partialCors <- c(0.1, 0.2)  # Wrong length (should be D-1 = 3)

  expect_error(
    updateVineMatrices(structureMatrix, familyMatrix, parameterMatrix, partialCors),
    "length D-1"
  )
})

test_that("updateVineMatrices correctly updates first column with partial correlations", {
  D <- 4
  structureMatrix <- matrix(c(
    4, 0, 0, 0,
    3, 4, 0, 0,
    2, 3, 4, 0,
    1, 2, 3, 4
  ), nrow = 4, byrow = TRUE)
  familyMatrix <- matrix(0, D, D)
  familyMatrix[D, 1:(D-1)] <- 3  # Clayton
  parameterMatrix <- matrix(0, D, D)
  parameterMatrix[D, 1:(D-1)] <- 2  # Clayton parameter
  partialCors <- c(0.1, 0.2, 0.3)

  result <- updateVineMatrices(structureMatrix, familyMatrix, parameterMatrix, partialCors)

  # First column (rows 2:D) should contain partial correlations (reversed)
  expect_equal(result$ParameterMatrix[D:2, 1], partialCors)
})

test_that("updateVineMatrices sets Gaussian family for first column", {
  D <- 4
  structureMatrix <- matrix(c(
    4, 0, 0, 0,
    3, 4, 0, 0,
    2, 3, 4, 0,
    1, 2, 3, 4
  ), nrow = 4, byrow = TRUE)
  familyMatrix <- matrix(0, D, D)
  familyMatrix[D, 1:(D-1)] <- 3  # Clayton
  parameterMatrix <- matrix(0, D, D)
  parameterMatrix[D, 1:(D-1)] <- 2
  partialCors <- c(0.1, 0.2, 0.3)

  result <- updateVineMatrices(structureMatrix, familyMatrix, parameterMatrix, partialCors)

  # First column (rows 2:D) should be Gaussian family (1)
  expect_equal(result$FamilyMatrix[2:D, 1], rep(1, D - 1))
})

test_that("updateVineMatrices handles 3-variable case", {
  D <- 3
  structureMatrix <- matrix(c(
    3, 0, 0,
    2, 3, 0,
    1, 2, 3
  ), nrow = 3, byrow = TRUE)
  familyMatrix <- matrix(0, D, D)
  familyMatrix[D, 1:(D-1)] <- 1
  parameterMatrix <- matrix(0, D, D)
  parameterMatrix[D, 1:(D-1)] <- 0.5
  partialCors <- c(0.3, 0.4)

  result <- updateVineMatrices(structureMatrix, familyMatrix, parameterMatrix, partialCors)

  expect_equal(dim(result$StructureMatrix), c(3, 3))
  expect_equal(result$ParameterMatrix[D:2, 1], partialCors)
})
