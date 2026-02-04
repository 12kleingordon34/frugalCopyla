# Integration test suite for the nonparanormal package
# Tests full pipeline functionality and Markov property preservation

test_that("full pipeline: covariate generation to outcome simulation", {
  skip_if_not_installed("copula")

  set.seed(42)
  n <- 500

  # Step 1: Generate covariates from a Bayesian Network
  U1 <- runif(n)
  Z1 <- qgamma(U1, shape = 2, scale = 2)

  U2_1 <- runif(n)
  Z2 <- qgamma(U2_1, shape = 2 + 1.5 * Z1, scale = 1)

  # Step 2: Prepare data for outcome generation
  covariate_data <- cbind(Z1, Z2)
  cond_covariate_ranks <- cbind(U1, U2_1)
  topoOrder <- c(2, 1)
  vine_cor_params <- c(0.5, 0.3)

  # Step 3: Generate outcome
  result <- simulateConditionalOutcomeSamples(
    covariate_data = covariate_data,
    cond_covariate_ranks = cond_covariate_ranks,
    vine_cor_params = vine_cor_params,
    topoOrder = topoOrder
  )

  # Verify output structure
  expect_true(!is.null(result$outcomeRankSamples))
  expect_length(result$outcomeRankSamples, n)
  expect_true(all(result$outcomeRankSamples >= 0 & result$outcomeRankSamples <= 1))
})

test_that("correlation structure preservation in pipeline", {
  skip_if_not_installed("copula")

  set.seed(42)
  n <- 2000

  # Generate covariates
  U1 <- runif(n)
  Z1 <- qnorm(U1)

  U2_1 <- runif(n)
  # Z2 | Z1 ~ N(0.5*Z1, sqrt(0.75))
  Z2 <- 0.5 * Z1 + sqrt(0.75) * qnorm(U2_1)

  covariate_data <- cbind(Z1, Z2)
  cond_covariate_ranks <- cbind(U1, U2_1)
  topoOrder <- c(2, 1)
  vine_cor_params <- c(0.6, 0.4)

  result <- simulateConditionalOutcomeSamples(
    covariate_data = pnorm(covariate_data),  # Convert to uniform
    cond_covariate_ranks = cond_covariate_ranks,
    vine_cor_params = vine_cor_params,
    topoOrder = topoOrder
  )

  # Check that full correlation matrix has the expected correlation with Z2
  full_cor <- result$fullCorrelationMatrix
  expect_true(abs(full_cor[2, 3] - 0.6) < 0.1)  # Z2-Y correlation
})

test_that("Markov property: conditional independence holds", {
  skip_if_not_installed("copula")
  skip_if_not_installed("GeneralisedCovarianceMeasure")
  skip("Long-running test - run manually")

  set.seed(42)
  n <- 5000

  # Generate T=2 time points
  # Time 1
  U1_1 <- runif(n)
  Z1_1 <- qgamma(U1_1, shape = 2, scale = 2)

  U2_1 <- runif(n)
  Z2_1 <- qgamma(U2_1, shape = 2 + 1.5 * Z1_1, scale = 1)

  # Time 2: depends on time 1
  U1_2 <- runif(n)
  Z1_2 <- qgamma(U1_2, shape = 2 + 1.5 * Z1_1, scale = 1)

  U2_2 <- runif(n)
  Z2_2 <- qgamma(U2_2, shape = 2 + Z1_2 + 0.5 * Z2_1, scale = 1)

  # Generate Y_2 based ONLY on Z^2 (not Z^1)
  covariate_data_t2 <- cbind(Z1_2, Z2_2)
  cond_covariate_ranks_t2 <- cbind(U1_2, U2_2)
  topoOrder <- c(2, 1)
  vine_cor_params <- c(0.5, 0.3)

  result <- simulateConditionalOutcomeSamples(
    covariate_data = covariate_data_t2,
    cond_covariate_ranks = cond_covariate_ranks_t2,
    vine_cor_params = vine_cor_params,
    topoOrder = topoOrder
  )

  Y_2 <- qnorm(result$outcomeRankSamples)

  # Scale variables
  Y_2_scaled <- scale(Y_2)
  Z1_1_scaled <- scale(Z1_1)
  Z1_2_scaled <- scale(Z1_2)
  Z2_2_scaled <- scale(Z2_2)

  # Test: Y_2 _|_ Z1_1 | (Z1_2, Z2_2)
  # This should hold by construction (Markov property)
  p_values <- bootstrappedCondIndTest_GCM(
    x = as.vector(Y_2_scaled),
    y = as.vector(Z1_1_scaled),
    z = cbind(Z1_2_scaled, Z2_2_scaled),
    n_boot = 100,
    sample_size = 2000,
    show_progress = FALSE,
    regr.method = 'gam'
  )

  # KS test for uniformity
  ks_result <- ks.test(p_values, "punif")
  expect_true(ks_result$p.value > 0.01)  # Should not reject uniformity
})

test_that("causal margin preservation", {
  skip_if_not_installed("copula")

  set.seed(42)
  n <- 3000

  # Simple case: Y = N(0, 1) marginally
  # Covariates are uniform
  covariate_data <- matrix(runif(n * 2), ncol = 2)
  marginal_ranks <- covariate_data
  topoOrder <- c(2, 1)
  vine_cor_params <- c(0.5, 0.3)

  result <- simulateMarginalOutcomeSamples(
    covariate_data = covariate_data,
    marginal_covariate_ranks = marginal_ranks,
    vine_cor_params = vine_cor_params,
    topoOrder = topoOrder
  )

  # Transform to Gaussian
  Y <- qnorm(result$outcomeRankSamples)

  # Y should be approximately N(0, 1) marginally
  # (The copula structure doesn't change the marginal)
  expect_true(abs(mean(Y)) < 0.1)
  expect_true(abs(sd(Y) - 1) < 0.1)

  # Shapiro-Wilk test for normality (on a subsample)
  sw_result <- shapiro.test(Y[1:min(5000, n)])
  expect_true(sw_result$p.value > 0.01)
})

test_that("pipeline handles different covariate dimensions", {
  skip_if_not_installed("copula")

  set.seed(42)

  for (d in 2:4) {
    n <- 500
    covariate_data <- matrix(runif(n * d), ncol = d)
    cond_ranks <- matrix(runif(n * d), ncol = d)
    topoOrder <- d:1
    vine_cor_params <- seq(0.5, 0.1, length.out = d)

    result <- simulateConditionalOutcomeSamples(
      covariate_data = covariate_data,
      cond_covariate_ranks = cond_ranks,
      vine_cor_params = vine_cor_params,
      topoOrder = topoOrder
    )

    expect_length(result$outcomeRankSamples, n)
    expect_equal(nrow(result$fullCorrelationMatrix), d + 1)
    expect_equal(ncol(result$fullCorrelationMatrix), d + 1)
  }
})

test_that("vine reparameterization produces valid simulations", {
  skip_if_not_installed("VineCopula")
  skip_if_not_installed("copula")

  set.seed(42)
  n <- 1000
  D <- 4

  # Original vine structure with Clayton copulas
  structureMatrix <- matrix(c(
    D, 0, 0, 0,
    D-1, D, 0, 0,
    D-2, D-1, D, 0,
    1, D-2, D-1, D
  ), nrow = D, byrow = TRUE)

  familyMatrix <- matrix(0, D, D)
  familyMatrix[D, 1:(D-1)] <- 3  # Clayton

  parameterMatrix <- matrix(0, D, D)
  parameterMatrix[D, 1:(D-1)] <- 2  # Clayton parameter

  topoOrder <- 1:(D-1)
  vineCorParams <- c(0.5, rep(0, D-2))

  result <- simulateAndReparameterizeVine(
    structureMatrix, familyMatrix, parameterMatrix,
    n, topoOrder, vineCorParams, seed = 42
  )

  # Check that new simulated data is valid
  expect_equal(nrow(result$newVineOutput$simdata), n)
  expect_equal(ncol(result$newVineOutput$simdata), D)
  expect_true(all(result$newVineOutput$simdata >= 0 &
                   result$newVineOutput$simdata <= 1))
})
