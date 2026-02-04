# Test suite for correlation computation functions

test_that("computeConditionalCovariance works for 1D conditioning", {
  # Simple case: no conditioning set
  rho <- 0.5
  Sigma_AB <- matrix(c(0.3, 0.4), nrow = 2)
  Sigma_BB <- 1

  result <- computeConditionalCovariance(rho, Sigma_AB, Sigma_BB)

  # The result should be a scalar
  expect_length(result, 1)
  expect_true(is.numeric(result))
})

test_that("computeConditionalCovariance works for multi-D conditioning", {
  rho <- 0.5
  Sigma_AB <- matrix(c(0.3, 0.2, 0.4, 0.3), nrow = 2)
  Sigma_BB <- matrix(c(1, 0.4, 0.4, 1), ncol = 2)

  result <- computeConditionalCovariance(rho, Sigma_AB, Sigma_BB)

  expect_length(result, 1)
  expect_true(is.numeric(result))
})

test_that("computeFullCorMatrix returns correct dimensions", {
  topoOrder <- c(2, 1)
  corMatrixMN <- matrix(c(1, 0.5, 0.5, 1), ncol = 2)
  vineCorParams <- c(0.6, 0.4)

  result <- computeFullCorMatrix(topoOrder, corMatrixMN, vineCorParams)

  # Should be (D+1) x (D+1) = 3x3
  expect_equal(dim(result), c(3, 3))
})

test_that("computeFullCorMatrix returns symmetric matrix", {
  topoOrder <- c(2, 1)
  corMatrixMN <- matrix(c(1, 0.5, 0.5, 1), ncol = 2)
  vineCorParams <- c(0.6, 0.4)

  result <- computeFullCorMatrix(topoOrder, corMatrixMN, vineCorParams)

  expect_equal(result, t(result))
})

test_that("computeFullCorMatrix has diagonal of ones", {
  topoOrder <- c(2, 1)
  corMatrixMN <- matrix(c(1, 0.5, 0.5, 1), ncol = 2)
  vineCorParams <- c(0.6, 0.4)

  result <- computeFullCorMatrix(topoOrder, corMatrixMN, vineCorParams)

  expect_equal(diag(result), rep(1, 3))
})

test_that("computeFullCorMatrix throws error for dimension mismatch", {
  topoOrder <- c(2, 1, 3)  # Length 3
  corMatrixMN <- matrix(c(1, 0.5, 0.5, 1), ncol = 2)  # 2x2

  expect_error(
    computeFullCorMatrix(topoOrder, corMatrixMN, c(0.6, 0.4)),
    "Dimension mismatch"
  )
})

test_that("computeFullCorMatrix places first vine param correctly", {
  # With topoOrder = c(2, 1), first param is correlation between Y and variable 2
  topoOrder <- c(2, 1)
  corMatrixMN <- matrix(c(1, 0.5, 0.5, 1), ncol = 2)
  vineCorParams <- c(0.6, 0.4)

  result <- computeFullCorMatrix(topoOrder, corMatrixMN, vineCorParams)

  # Variable 2 (row 2) should have correlation 0.6 with Y (row 3)
  expect_equal(result[2, 3], 0.6)
  expect_equal(result[3, 2], 0.6)
})

test_that("computePartialCorrelations returns correct length", {
  # Create a known correlation matrix
  fullCorMatrix <- matrix(c(
    1.0, 0.5, 0.3,
    0.5, 1.0, 0.4,
    0.3, 0.4, 1.0
  ), ncol = 3, byrow = TRUE)

  # topoOrder should include Y index as last element
  topoOrder <- c(1, 2, 3)  # Variables 1, 2, then Y (index 3)

  partials <- computePartialCorrelations(fullCorMatrix, topoOrder)

  # Should return D-1 = 2 partial correlations
  expect_length(partials, 2)
  expect_true(all(abs(partials) <= 1))  # Valid correlation range
})

test_that("calculateSequentialPartialCorrelations works correctly", {
  # 3x3 correlation matrix
  corMatrix <- matrix(c(
    1.0, 0.5, 0.3,
    0.5, 1.0, 0.4,
    0.3, 0.4, 1.0
  ), ncol = 3, byrow = TRUE)

  partials <- calculateSequentialPartialCorrelations(corMatrix)

  expect_length(partials, 2)
  expect_true(all(abs(partials) <= 1))
})

test_that("calculateSequentialPartialCorrelations requires square matrix", {
  nonsquare <- matrix(1:6, ncol = 2)

  expect_error(
    calculateSequentialPartialCorrelations(nonsquare),
    "square matrix"
  )
})

test_that("compute_standardized_precision_matrix returns correct structure", {
  set.seed(123)
  # Generate uniform data
  data <- matrix(runif(300), ncol = 3)

  result <- compute_standardized_precision_matrix(data)

  # Should be square
  expect_equal(nrow(result), ncol(result))
  expect_equal(nrow(result), ncol(data))

  # Diagonal should be 1 (standardized)
  expect_equal(diag(result), rep(1, ncol(data)), tolerance = 1e-10)

  # Should be symmetric
  expect_equal(result, t(result), tolerance = 1e-10)
})

test_that("is.square.matrix works correctly", {
  square <- matrix(1:9, ncol = 3)
  nonsquare <- matrix(1:6, ncol = 2)

  expect_true(is.square.matrix(square))
  expect_false(is.square.matrix(nonsquare))
})

test_that("computeFullCorMatrix produces valid correlation matrix", {
  topoOrder <- c(2, 1)
  corMatrixMN <- matrix(c(1, 0.5, 0.5, 1), ncol = 2)
  vineCorParams <- c(0.6, 0.3)

  result <- computeFullCorMatrix(topoOrder, corMatrixMN, vineCorParams)

  # Check positive definiteness
  eigenvalues <- eigen(result)$values
  expect_true(all(eigenvalues > 0))

  # All correlations should be in [-1, 1]
  expect_true(all(result >= -1 & result <= 1))
})
