# Test suite for rank transformation functions

test_that("uncondition_conditional_ranks returns correct dimensions", {
  set.seed(123)
  n <- 100
  d <- 3
  cond_ranks <- matrix(runif(n * d), ncol = d)
  R <- matrix(c(1, 0.5, 0.3, 0.5, 1, 0.4, 0.3, 0.4, 1), ncol = 3, byrow = TRUE)

  result <- uncondition_conditional_ranks(cond_ranks, R)

  expect_equal(dim(result), c(n, d))
})

test_that("uncondition_conditional_ranks output is in [0, 1]", {
  set.seed(123)
  n <- 200
  d <- 3
  cond_ranks <- matrix(runif(n * d), ncol = d)
  R <- matrix(c(1, 0.5, 0.3, 0.5, 1, 0.4, 0.3, 0.4, 1), ncol = 3, byrow = TRUE)

  result <- uncondition_conditional_ranks(cond_ranks, R)

  expect_true(all(result >= 0 & result <= 1))
})

test_that("uncondition_conditional_ranks first column unchanged when properly conditioned", {
  set.seed(123)
  n <- 100
  d <- 3
  cond_ranks <- matrix(runif(n * d), ncol = d)
  R <- matrix(c(1, 0.5, 0.3, 0.5, 1, 0.4, 0.3, 0.4, 1), ncol = 3, byrow = TRUE)

  result <- uncondition_conditional_ranks(cond_ranks, R)

  # First column should remain uniform (it's unconditional)
  # Test via KS test for uniformity
  ks_result <- ks.test(result[, 1], "punif")
  expect_true(ks_result$p.value > 0.01)  # Should not reject uniformity
})

test_that("uncondition_conditional_ranks handles 2-variable case", {
  set.seed(123)
  n <- 100
  cond_ranks <- matrix(runif(n * 2), ncol = 2)
  R <- matrix(c(1, 0.6, 0.6, 1), ncol = 2)

  result <- uncondition_conditional_ranks(cond_ranks, R)

  expect_equal(dim(result), c(n, 2))
  expect_true(all(result >= 0 & result <= 1))
})

test_that("uncondition_conditional_ranks with identity correlation matrix", {
  set.seed(123)
  n <- 100
  d <- 3
  cond_ranks <- matrix(runif(n * d), ncol = d)
  R <- diag(d)  # Identity = no correlation

  result <- uncondition_conditional_ranks(cond_ranks, R)

  # With identity correlation, conditional = marginal
  # So the result should be the same as input (after transformation)
  expect_equal(dim(result), c(n, d))
})

test_that("uncondition_conditional_ranks produces uniform marginals for Gaussian case", {
  skip_if_not_installed("MASS")

  set.seed(123)
  n <- 1000

  # Generate from a true Gaussian copula
  R <- matrix(c(1, 0.5, 0.3, 0.5, 1, 0.4, 0.3, 0.4, 1), ncol = 3, byrow = TRUE)
  Z <- MASS::mvrnorm(n, mu = rep(0, 3), Sigma = R)

  # Convert to uniform marginals
  U <- pnorm(Z)

  # Create conditional ranks from the Gaussian structure
  # U1 is marginal, U2|1 is conditional, U3|1,2 is conditional
  # For Gaussian: U_{j|1:(j-1)} = Phi((Z_j - mu_{j|1:(j-1)}) / sigma_{j|1:(j-1)})
  cond_ranks <- matrix(NA, n, 3)
  cond_ranks[, 1] <- U[, 1]  # First is marginal

  # Z2 | Z1: mu = R[2,1] * Z1, sigma^2 = 1 - R[2,1]^2
  mu2 <- R[2, 1] * Z[, 1]
  sigma2 <- sqrt(1 - R[2, 1]^2)
  cond_ranks[, 2] <- pnorm((Z[, 2] - mu2) / sigma2)

  # Z3 | Z1, Z2
  R_12 <- matrix(R[3, 1:2], nrow = 1)
  R_22 <- R[1:2, 1:2]
  mu3 <- as.vector(R_12 %*% solve(R_22) %*% t(Z[, 1:2]))
  sigma3 <- sqrt(1 - as.numeric(R_12 %*% solve(R_22) %*% t(R_12)))
  cond_ranks[, 3] <- pnorm((Z[, 3] - mu3) / sigma3)

  # Now uncondition
  result <- uncondition_conditional_ranks(cond_ranks, R)

  # Result should be close to original marginal uniforms U
  # Use unname to ignore dimnames differences
  expect_equal(unname(result), unname(U), tolerance = 1e-6)
})

test_that("uncondition_conditional_ranks handles data frame input", {
  set.seed(123)
  n <- 50
  cond_ranks <- data.frame(V1 = runif(n), V2 = runif(n), V3 = runif(n))
  R <- matrix(c(1, 0.5, 0.3, 0.5, 1, 0.4, 0.3, 0.4, 1), ncol = 3, byrow = TRUE)

  result <- uncondition_conditional_ranks(cond_ranks, R)

  expect_equal(dim(result), c(n, 3))
  expect_true(is.matrix(result))
})

# =============================================================================
# DAG-aware unconditioning tests
# =============================================================================

test_that("uncondition_conditional_ranks accepts parents argument", {
  set.seed(123)
  n <- 100
  cond_ranks <- matrix(runif(n * 3), ncol = 3)
  R <- matrix(c(1, 0.5, 0.3, 0.5, 1, 0.4, 0.3, 0.4, 1), ncol = 3, byrow = TRUE)

  # Chain DAG: Z1 -> Z2 -> Z3
  parents <- list(integer(0), c(1), c(2))

  result <- uncondition_conditional_ranks(cond_ranks, R, parents)

  expect_equal(dim(result), c(n, 3))
  expect_true(all(result >= 0 & result <= 1))
})

test_that("DAG parents produce different results than D-vine", {
  set.seed(456)
  n <- 500
  cond_ranks <- matrix(runif(n * 3), ncol = 3)
  R <- matrix(c(1, 0.5, 0.3, 0.5, 1, 0.4, 0.3, 0.4, 1), ncol = 3, byrow = TRUE)

  # D-vine (default): Z3 | Z1, Z2
  result_dvine <- uncondition_conditional_ranks(cond_ranks, R)

  # Chain: Z3 | Z2 only
  parents_chain <- list(integer(0), c(1), c(2))
  result_chain <- uncondition_conditional_ranks(cond_ranks, R, parents_chain)

  # Results should differ (especially column 3)
  # Column 1 and 2 should be the same
  expect_equal(result_dvine[, 1], result_chain[, 1])
  expect_equal(result_dvine[, 2], result_chain[, 2])

  # Column 3 should be different
  expect_false(all(abs(result_dvine[, 3] - result_chain[, 3]) < 1e-10))
})

test_that("uncondition_conditional_ranks validates topological order", {
  set.seed(123)
  cond_ranks <- matrix(runif(30), ncol = 3)
  R <- matrix(c(1, 0.5, 0.3, 0.5, 1, 0.4, 0.3, 0.4, 1), ncol = 3, byrow = TRUE)

  # Invalid: variable 1 depends on variable 2
  bad_parents <- list(c(2), integer(0), c(1))

  expect_error(
    uncondition_conditional_ranks(cond_ranks, R, bad_parents),
    "Invalid topological order"
  )
})

test_that("uncondition_conditional_ranks check_order=FALSE skips validation", {
  set.seed(123)
  cond_ranks <- matrix(runif(30), ncol = 3)
  R <- matrix(c(1, 0.5, 0.3, 0.5, 1, 0.4, 0.3, 0.4, 1), ncol = 3, byrow = TRUE)

  # Invalid order but skip check (will produce wrong results but no error)
  bad_parents <- list(c(2), integer(0), c(1))

  # Should not error when check_order = FALSE
  # (Note: results will be wrong, but we're testing that the check is skipped)
  expect_error(
    uncondition_conditional_ranks(cond_ranks, R, bad_parents, check_order = FALSE),
    NA
  )
})

test_that("DAG-aware unconditioning works for longitudinal structure", {
  skip_if_not_installed("MASS")

  set.seed(789)
  n <- 1000

  # 2 time points, 2 covariates = 4 columns
  # Order: Z1_1, Z2_1, Z1_2, Z2_2
  d <- 4

  # Build a correlation matrix consistent with Markov structure
  R <- matrix(c(
    1.0, 0.4, 0.6, 0.2,  # Z1_1
    0.4, 1.0, 0.3, 0.5,  # Z2_1
    0.6, 0.3, 1.0, 0.4,  # Z1_2
    0.2, 0.5, 0.4, 1.0   # Z2_2
  ), ncol = 4, byrow = TRUE)

  # Generate from true Gaussian
  Z <- MASS::mvrnorm(n, mu = rep(0, d), Sigma = R)

  # Create conditional ranks from Markov DAG structure:
  # Z1_1: root
  # Z2_1 | Z1_1
  # Z1_2 | Z1_1 (NOT Z2_1!)
  # Z2_2 | Z2_1, Z1_2
  parents <- list(
    integer(0),   # Z1_1
    c(1),         # Z2_1 | Z1_1
    c(1),         # Z1_2 | Z1_1
    c(2, 3)       # Z2_2 | Z2_1, Z1_2
  )

  cond_ranks <- matrix(NA, n, d)

  # Z1_1: marginal
  cond_ranks[, 1] <- pnorm(Z[, 1])

  # Z2_1 | Z1_1
  mu_2 <- R[2, 1] * Z[, 1]
  sigma_2 <- sqrt(1 - R[2, 1]^2)
  cond_ranks[, 2] <- pnorm((Z[, 2] - mu_2) / sigma_2)

  # Z1_2 | Z1_1
  mu_3 <- R[3, 1] * Z[, 1]
  sigma_3 <- sqrt(1 - R[3, 1]^2)
  cond_ranks[, 3] <- pnorm((Z[, 3] - mu_3) / sigma_3)

  # Z2_2 | Z2_1, Z1_2
  pa_4 <- c(2, 3)
  r_vec <- matrix(R[4, pa_4], nrow = 1)
  R_sub <- R[pa_4, pa_4]
  R_sub_inv <- solve(R_sub)
  mu_4 <- as.vector(Z[, pa_4] %*% R_sub_inv %*% t(r_vec))
  sigma_4 <- sqrt(1 - as.numeric(r_vec %*% R_sub_inv %*% t(r_vec)))
  cond_ranks[, 4] <- pnorm((Z[, 4] - mu_4) / sigma_4)

  # Now uncondition
  result <- uncondition_conditional_ranks(cond_ranks, R, parents)

  # Should recover original marginal uniforms
  U_original <- pnorm(Z)
  expect_equal(unname(result), unname(U_original), tolerance = 1e-6)
})

test_that("uncondition_conditional_ranks handles single parent correctly", {
  set.seed(999)
  n <- 500

  R <- matrix(c(1.0, 0.7, 0.7, 1.0), ncol = 2)

  # Generate true Gaussian
  Z <- MASS::mvrnorm(n, mu = c(0, 0), Sigma = R)

  # Create conditional ranks
  cond_ranks <- matrix(NA, n, 2)
  cond_ranks[, 1] <- pnorm(Z[, 1])

  mu_2 <- R[2, 1] * Z[, 1]
  sigma_2 <- sqrt(1 - R[2, 1]^2)
  cond_ranks[, 2] <- pnorm((Z[, 2] - mu_2) / sigma_2)

  parents <- list(integer(0), c(1))

  result <- uncondition_conditional_ranks(cond_ranks, R, parents)

  # Should recover original
  U_original <- pnorm(Z)
  expect_equal(unname(result), unname(U_original), tolerance = 1e-6)
})
