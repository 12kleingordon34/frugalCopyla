# Test suite for outcome generation functions

test_that("multivariate_conditional_mean_and_samples returns correct structure", {
  set.seed(123)
  n <- 100
  X2 <- matrix(rnorm(n * 2), ncol = 2)
  R <- matrix(c(1, 0.5, 0.3, 0.5, 1, 0.4, 0.3, 0.4, 1), ncol = 3, byrow = TRUE)

  result <- multivariate_conditional_mean_and_samples(X2, R)

  expect_true("generated_samples" %in% names(result))
  expect_true("conditional_means" %in% names(result))
  expect_true("conditional_variance" %in% names(result))
  expect_true("coeffs" %in% names(result))
})

test_that("multivariate_conditional_mean_and_samples returns correct dimensions", {
  set.seed(123)
  n <- 100
  k <- 3  # Conditioning on 3 variables
  X2 <- matrix(rnorm(n * k), ncol = k)
  R <- diag(k + 1)
  R[1:k, k + 1] <- R[k + 1, 1:k] <- c(0.3, 0.4, 0.2)
  R[1, 2] <- R[2, 1] <- 0.5
  R[1, 3] <- R[3, 1] <- 0.3
  R[2, 3] <- R[3, 2] <- 0.4

  result <- multivariate_conditional_mean_and_samples(X2, R)

  expect_equal(nrow(result$generated_samples), n)
  expect_equal(ncol(result$generated_samples), 1)
  expect_equal(length(result$conditional_means), n)
})

test_that("multivariate_conditional_mean_and_samples conditional variance is positive", {
  set.seed(123)
  X2 <- matrix(rnorm(200), ncol = 2)
  R <- matrix(c(1, 0.5, 0.3, 0.5, 1, 0.4, 0.3, 0.4, 1), ncol = 3, byrow = TRUE)

  result <- multivariate_conditional_mean_and_samples(X2, R)

  expect_true(result$conditional_variance > 0)
})

test_that("simulateMarginalOutcomeSamples returns correct structure", {
  skip_if_not_installed("copula")

  set.seed(123)
  n <- 200
  cov_data <- matrix(runif(n * 2), ncol = 2)
  marginal_ranks <- cov_data
  vine_cor_params <- c(0.5, 0.3)
  topoOrder <- c(2, 1)

  result <- simulateMarginalOutcomeSamples(cov_data, marginal_ranks,
                                            vine_cor_params, topoOrder)

  expect_true("gaussianCopulaFit" %in% names(result))
  expect_true("fullCorrelationMatrix" %in% names(result))
  expect_true("outcomeRankSamples" %in% names(result))
})

test_that("simulateMarginalOutcomeSamples outcome is in [0, 1]", {
  skip_if_not_installed("copula")

  set.seed(123)
  n <- 200
  cov_data <- matrix(runif(n * 2), ncol = 2)
  marginal_ranks <- cov_data
  vine_cor_params <- c(0.5, 0.3)
  topoOrder <- c(2, 1)

  result <- simulateMarginalOutcomeSamples(cov_data, marginal_ranks,
                                            vine_cor_params, topoOrder)

  expect_true(all(result$outcomeRankSamples >= 0 &
                   result$outcomeRankSamples <= 1))
})

test_that("simulateConditionalOutcomeSamples returns correct structure", {
  skip_if_not_installed("copula")

  set.seed(123)
  n <- 200
  cov_data <- matrix(runif(n * 2), ncol = 2)
  cond_ranks <- matrix(runif(n * 2), ncol = 2)
  vine_cor_params <- c(0.5, 0.3)
  topoOrder <- c(2, 1)

  result <- simulateConditionalOutcomeSamples(cov_data, cond_ranks,
                                               vine_cor_params, topoOrder)

  expect_true("gaussianCopulaFit" %in% names(result))
  expect_true("fullCorrelationMatrix" %in% names(result))
  expect_true("outcomeRankSamples" %in% names(result))
})

test_that("simulateConditionalOutcomeSamples outcome is in [0, 1]", {
  skip_if_not_installed("copula")

  set.seed(123)
  n <- 200
  cov_data <- matrix(runif(n * 2), ncol = 2)
  cond_ranks <- matrix(runif(n * 2), ncol = 2)
  vine_cor_params <- c(0.5, 0.3)
  topoOrder <- c(2, 1)

  result <- simulateConditionalOutcomeSamples(cov_data, cond_ranks,
                                               vine_cor_params, topoOrder)

  expect_true(all(result$outcomeRankSamples >= 0 &
                   result$outcomeRankSamples <= 1))
})

test_that("simulateConditionalOutcomeSamples handles default topoOrder", {
  skip_if_not_installed("copula")

  set.seed(123)
  n <- 200
  d <- 2
  cov_data <- matrix(runif(n * d), ncol = d)
  cond_ranks <- matrix(runif(n * d), ncol = d)
  # For default topoOrder (1:d), need d params
  vine_cor_params <- c(0.5, 0.3)

  result <- simulateConditionalOutcomeSamples(cov_data, cond_ranks,
                                               vine_cor_params, topoOrder = NULL)

  expect_equal(length(result$outcomeRankSamples), n)
})

test_that("simulateMarginalOutcomeSamples correlation structure is preserved", {
  skip_if_not_installed("copula")

  set.seed(123)
  n <- 2000

  # Generate uniform covariates
  cov_data <- matrix(runif(n * 2), ncol = 2)
  marginal_ranks <- cov_data
  vine_cor_params <- c(0.6, 0.0)  # Strong correlation with first covariate only
  topoOrder <- c(2, 1)

  result <- simulateMarginalOutcomeSamples(cov_data, marginal_ranks,
                                            vine_cor_params, topoOrder)

  # Check correlation with first covariate in topo order (which is column 2)
  # The full correlation matrix should have the specified structure
  full_cor <- result$fullCorrelationMatrix
  expect_true(abs(full_cor[2, 3] - 0.6) < 0.05)  # Specified correlation
})
