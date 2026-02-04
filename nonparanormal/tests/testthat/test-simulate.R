# Test suite for simulation functions

test_that("simulateRVineData returns correct dimensions", {
  skip_if_not_installed("VineCopula")

  # Valid D-vine structure for 3 variables
  structureMatrix <- matrix(c(
    1, 0, 0,
    2, 2, 0,
    3, 3, 3
  ), ncol = 3, byrow = TRUE)

  # Independence copula (family 0)
  familyMatrix <- matrix(c(
    0, 0, 0,
    0, 0, 0,
    0, 0, 0
  ), ncol = 3, byrow = TRUE)

  parameterMatrix <- matrix(c(
    0, 0, 0,
    0, 0, 0,
    0, 0, 0
  ), ncol = 3, byrow = TRUE)

  result <- simulateRVineData(structureMatrix, familyMatrix, parameterMatrix,
                               sampleSize = 100, seed = 123)

  expect_equal(nrow(result$simdata), 100)
  expect_equal(ncol(result$simdata), 3)
  expect_true(all(result$simdata >= 0 & result$simdata <= 1))
})

test_that("simulateRVineData is reproducible with same seed", {
  skip_if_not_installed("VineCopula")

  structureMatrix <- matrix(c(
    1, 0, 0,
    2, 2, 0,
    3, 3, 3
  ), ncol = 3, byrow = TRUE)

  familyMatrix <- matrix(0, 3, 3)
  parameterMatrix <- matrix(0, 3, 3)

  result1 <- simulateRVineData(structureMatrix, familyMatrix, parameterMatrix,
                                sampleSize = 50, seed = 42)
  result2 <- simulateRVineData(structureMatrix, familyMatrix, parameterMatrix,
                                sampleSize = 50, seed = 42)

  expect_equal(result1$simdata, result2$simdata)
})

test_that("simulateRVineData produces different results with different seeds", {
  skip_if_not_installed("VineCopula")

  structureMatrix <- matrix(c(
    1, 0, 0,
    2, 2, 0,
    3, 3, 3
  ), ncol = 3, byrow = TRUE)

  familyMatrix <- matrix(0, 3, 3)
  parameterMatrix <- matrix(0, 3, 3)

  result1 <- simulateRVineData(structureMatrix, familyMatrix, parameterMatrix,
                                sampleSize = 50, seed = 42)
  result2 <- simulateRVineData(structureMatrix, familyMatrix, parameterMatrix,
                                sampleSize = 50, seed = 43)

  expect_false(all(result1$simdata == result2$simdata))
})

test_that("simulateRVineData returns RVM object", {
  skip_if_not_installed("VineCopula")

  structureMatrix <- matrix(c(
    1, 0, 0,
    2, 2, 0,
    3, 3, 3
  ), ncol = 3, byrow = TRUE)

  familyMatrix <- matrix(0, 3, 3)
  parameterMatrix <- matrix(0, 3, 3)

  result <- simulateRVineData(structureMatrix, familyMatrix, parameterMatrix,
                               sampleSize = 50, seed = 123)

  expect_true("RVM" %in% names(result))
  expect_true("simdata" %in% names(result))
  # RVineMatrix may be S4 or S3 depending on VineCopula version
  expect_true(inherits(result$RVM, "RVineMatrix"))
})

test_that("simulateRVineData handles 2-variable case", {
  skip_if_not_installed("VineCopula")

  # 2-variable case (minimal)
  structureMatrix <- matrix(c(
    1, 0,
    2, 2
  ), ncol = 2, byrow = TRUE)

  familyMatrix <- matrix(0, 2, 2)
  parameterMatrix <- matrix(0, 2, 2)

  result <- simulateRVineData(structureMatrix, familyMatrix, parameterMatrix,
                               sampleSize = 100, seed = 123)

  expect_equal(ncol(result$simdata), 2)
  expect_equal(nrow(result$simdata), 100)
})

test_that("simulateRVineData with Gaussian copula produces correlations", {
  skip_if_not_installed("VineCopula")

  # 3-variable with Gaussian copula on the first pair
  structureMatrix <- matrix(c(
    1, 0, 0,
    2, 2, 0,
    3, 3, 3
  ), ncol = 3, byrow = TRUE)

  # Family 1 = Gaussian
  familyMatrix <- matrix(c(
    0, 0, 0,
    0, 0, 0,
    1, 0, 0
  ), ncol = 3, byrow = TRUE)

  # High correlation
  parameterMatrix <- matrix(c(
    0, 0, 0,
    0, 0, 0,
    0.8, 0, 0
  ), ncol = 3, byrow = TRUE)

  result <- simulateRVineData(structureMatrix, familyMatrix, parameterMatrix,
                               sampleSize = 1000, seed = 123)

  # Check that variables 1 and 3 are correlated (via variable 2)
  emp_cor <- cor(result$simdata[, 1], result$simdata[, 3])
  expect_true(abs(emp_cor) > 0.5)  # Should be positively correlated
})
