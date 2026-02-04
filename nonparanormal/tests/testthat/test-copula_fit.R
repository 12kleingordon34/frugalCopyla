# Test suite for copula fitting functions

test_that("fitMVGaussianCopula returns correct structure", {
  skip_if_not_installed("copula")

  set.seed(123)
  # Generate uniform data with known correlation structure
  n <- 500
  data <- matrix(runif(n * 3), ncol = 3)

  result <- fitMVGaussianCopula(data, method = 'itau')

  expect_true("gaussCop" %in% names(result))
  expect_true("fit" %in% names(result))
  expect_true("correlationMatrix" %in% names(result))
  expect_true("stdErrorMatrix" %in% names(result))
})

test_that("fitMVGaussianCopula returns symmetric correlation matrix", {
  skip_if_not_installed("copula")

  set.seed(123)
  n <- 500
  data <- matrix(runif(n * 3), ncol = 3)

  result <- fitMVGaussianCopula(data, method = 'itau')
  corMat <- result$correlationMatrix

  # Check symmetry
  expect_equal(corMat, t(corMat))

  # Check diagonal is all ones
  expect_equal(diag(corMat), rep(1, ncol(data)))
})

test_that("fitMVGaussianCopula returns positive definite matrix", {
  skip_if_not_installed("copula")

  set.seed(123)
  n <- 500
  data <- matrix(runif(n * 3), ncol = 3)

  result <- fitMVGaussianCopula(data, method = 'itau')
  corMat <- result$correlationMatrix

  # Check positive definiteness (all eigenvalues > 0)
  eigenvalues <- eigen(corMat)$values
  expect_true(all(eigenvalues > 0))
})

test_that("fitMVGaussianCopula correlation matrix has correct dimensions", {
  skip_if_not_installed("copula")

  set.seed(123)
  for (d in 2:5) {
    data <- matrix(runif(500 * d), ncol = d)
    result <- fitMVGaussianCopula(data, method = 'itau')

    expect_equal(nrow(result$correlationMatrix), d)
    expect_equal(ncol(result$correlationMatrix), d)
    expect_equal(nrow(result$stdErrorMatrix), d)
    expect_equal(ncol(result$stdErrorMatrix), d)
  }
})

test_that("fitMVGaussianCopula captures known correlation", {
  skip_if_not_installed("copula")
  skip_if_not_installed("MASS")

  set.seed(123)
  n <- 2000

  # Generate correlated normal data, then transform to uniform
  true_cor <- 0.6
  Sigma <- matrix(c(1, true_cor, true_cor, 1), ncol = 2)
  normal_data <- MASS::mvrnorm(n, mu = c(0, 0), Sigma = Sigma)
  uniform_data <- pnorm(normal_data)

  result <- fitMVGaussianCopula(uniform_data, method = 'itau')
  estimated_cor <- result$correlationMatrix[1, 2]

  # Should be close to true correlation (within 0.1)
  expect_true(abs(estimated_cor - true_cor) < 0.1)
})

test_that("fitMVGaussianCopula works with different methods", {
  skip_if_not_installed("copula")

  set.seed(123)
  n <- 300
  data <- matrix(runif(n * 2), ncol = 2)

  # Test itau method
  result_itau <- fitMVGaussianCopula(data, method = 'itau')
  expect_true(!is.null(result_itau$correlationMatrix))

  # Test mpl method (may be slower)
  result_mpl <- fitMVGaussianCopula(data, method = 'mpl')
  expect_true(!is.null(result_mpl$correlationMatrix))
})
