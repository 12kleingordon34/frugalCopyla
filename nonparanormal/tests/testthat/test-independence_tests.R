# Test suite for independence testing functions

test_that("bootstrappedKendallTest returns correct length", {
  set.seed(123)
  n <- 200
  x <- rnorm(n)
  y <- rnorm(n)

  p_values <- bootstrappedKendallTest(x, y, n_boot = 50, sample_size = 100,
                                       show_progress = FALSE)

  expect_length(p_values, 50)
})

test_that("bootstrappedKendallTest p-values are in [0, 1]", {
  set.seed(123)
  n <- 200
  x <- rnorm(n)
  y <- rnorm(n)

  p_values <- bootstrappedKendallTest(x, y, n_boot = 50, sample_size = 100,
                                       show_progress = FALSE)

  expect_true(all(p_values >= 0 & p_values <= 1))
})

test_that("bootstrappedKendallTest p-values are uniform under independence", {
  set.seed(42)
  n <- 1000
  x <- rnorm(n)
  y <- rnorm(n)  # Independent of x

  p_values <- bootstrappedKendallTest(x, y, n_boot = 100, sample_size = 500,
                                       show_progress = FALSE)

  # KS test for uniformity - should not reject
  ks_result <- ks.test(p_values, "punif")
  expect_true(ks_result$p.value > 0.01)
})

test_that("bootstrappedKendallTest p-values are non-uniform under dependence", {
  set.seed(42)
  n <- 1000
  x <- rnorm(n)
  y <- 0.8 * x + 0.2 * rnorm(n)  # Strongly dependent

  p_values <- bootstrappedKendallTest(x, y, n_boot = 100, sample_size = 500,
                                       show_progress = FALSE)

  # Most p-values should be small
  expect_true(mean(p_values < 0.05) > 0.9)
})

test_that("bootstrappedKendallTest throws error for unequal lengths", {
  expect_error(
    bootstrappedKendallTest(1:10, 1:5, show_progress = FALSE),
    "same length"
  )
})

test_that("bootstrappedCondIndTest_GCM returns correct length", {
  skip_if_not_installed("GeneralisedCovarianceMeasure")

  set.seed(123)
  n <- 200
  Z <- rnorm(n)
  X <- Z + rnorm(n)
  Y <- Z + rnorm(n)

  p_values <- bootstrappedCondIndTest_GCM(X, Y, matrix(Z, ncol = 1),
                                           n_boot = 20, sample_size = 100,
                                           show_progress = FALSE)

  expect_length(p_values, 20)
})

test_that("bootstrappedCondIndTest_GCM p-values are in [0, 1]", {
  skip_if_not_installed("GeneralisedCovarianceMeasure")

  set.seed(123)
  n <- 200
  Z <- rnorm(n)
  X <- Z + rnorm(n)
  Y <- Z + rnorm(n)

  p_values <- bootstrappedCondIndTest_GCM(X, Y, matrix(Z, ncol = 1),
                                           n_boot = 20, sample_size = 100,
                                           show_progress = FALSE)

  # Remove NAs if any
  p_values <- p_values[!is.na(p_values)]
  expect_true(all(p_values >= 0 & p_values <= 1))
})

test_that("bootstrappedCondIndTest_GCM throws error for dimension mismatch", {
  skip_if_not_installed("GeneralisedCovarianceMeasure")

  expect_error(
    bootstrappedCondIndTest_GCM(1:10, 1:5, matrix(1:10, ncol = 1),
                                 show_progress = FALSE),
    "same length"
  )

  expect_error(
    bootstrappedCondIndTest_GCM(1:10, 1:10, matrix(1:5, ncol = 1),
                                 show_progress = FALSE),
    "rows in z"
  )
})

test_that("bootstrappedCITest returns correct length", {
  skip_if_not_installed("bnlearn")

  set.seed(123)
  n <- 200
  x <- rnorm(n)
  y <- rnorm(n)
  z <- data.frame(Z1 = rnorm(n))

  p_values <- bootstrappedCITest(x, y, z, n_boot = 20, sample_size = 100,
                                  test = "cor", show_progress = FALSE)

  expect_length(p_values, 20)
})

test_that("bootstrappedKCI returns correct length", {
  skip_if_not_installed("CondIndTests")

  set.seed(123)
  n <- 100
  Z <- rnorm(n)
  X <- Z + rnorm(n)
  Y <- Z + rnorm(n)

  # Use small n_boot due to computational cost
  p_values <- bootstrappedKCI(X, Y, matrix(Z, ncol = 1),
                               n_boot = 5, sample_size = 50,
                               show_progress = FALSE)

  expect_length(p_values, 5)
})

test_that("bootstrappedCondIndTest_CIT returns correct length", {
  skip_if_not_installed("CondIndTests")

  set.seed(123)
  n <- 100
  Z <- rnorm(n)
  X <- Z + rnorm(n)
  Y <- Z + rnorm(n)

  # Use small n_boot due to computational cost
  p_values <- bootstrappedCondIndTest_CIT(X, Y, matrix(Z, ncol = 1),
                                           n_boot = 5, sample_size = 50,
                                           show_progress = FALSE)

  expect_length(p_values, 5)
})

test_that("bootstrap tests handle multiple conditioning variables", {
  skip_if_not_installed("GeneralisedCovarianceMeasure")

  set.seed(123)
  n <- 200
  Z1 <- rnorm(n)
  Z2 <- rnorm(n)
  X <- Z1 + Z2 + rnorm(n)
  Y <- Z1 + Z2 + rnorm(n)

  Z <- cbind(Z1, Z2)

  p_values <- bootstrappedCondIndTest_GCM(X, Y, Z,
                                           n_boot = 10, sample_size = 100,
                                           show_progress = FALSE)

  expect_length(p_values, 10)
  p_values_clean <- p_values[!is.na(p_values)]
  expect_true(all(p_values_clean >= 0 & p_values_clean <= 1))
})
