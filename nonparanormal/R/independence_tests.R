#' @title Independence Testing Functions
#' @description Functions for bootstrapped conditional and unconditional
#'   independence tests.
#' @name independence_tests
NULL

#' Bootstrapped Kendall's Tau Test
#'
#' Performs a bootstrapped hypothesis test using Kendall's tau on two input vectors.
#' For each bootstrap iteration, it resamples (with replacement) the data and computes
#' the Kendall's tau test p-value.
#'
#' Under the null hypothesis of independence, the distribution of p-values should
#' be approximately uniform on [0, 1].
#'
#' @param x A numeric vector.
#' @param y A numeric vector of the same length as x.
#' @param n_boot Integer. The number of bootstrap iterations. Default is 500.
#' @param sample_size Integer. The number of samples to draw in each bootstrap
#'   iteration. Default is the length of x.
#' @param show_progress Logical. Whether to show a progress bar. Default is TRUE.
#'
#' @return A numeric vector of bootstrapped p-values from Kendall's tau tests.
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' x <- rnorm(1000)
#' y <- rnorm(1000)
#' p_values <- bootstrappedKendallTest(x, y, n_boot = 100)
#' hist(p_values)
#' }
#'
#' @importFrom stats cor.test
#' @importFrom utils setTxtProgressBar txtProgressBar
#' @export
bootstrappedKendallTest <- function(x, y, n_boot = 500, sample_size = length(x),
                                     show_progress = TRUE) {
  if (length(x) != length(y)) {
    stop("x and y must be of the same length")
  }

  n <- length(x)
  p_values <- numeric(n_boot)

  if (show_progress) {
    pb <- utils::txtProgressBar(min = 0, max = n_boot, style = 3)
  }

  for (i in 1:n_boot) {
    boot_idx <- sample(1:n, sample_size, replace = TRUE)
    boot_x <- x[boot_idx]
    boot_y <- y[boot_idx]

    test_result <- stats::cor.test(boot_x, boot_y, method = "kendall")
    p_values[i] <- test_result$p.value

    if (show_progress) {
      utils::setTxtProgressBar(pb, i)
    }
  }

  if (show_progress) {
    close(pb)
  }

  return(p_values)
}


#' Bootstrapped Kernel Conditional Independence Test (KCI)
#'
#' Performs a bootstrapped conditional independence test using the KCI test.
#' It resamples the data (with replacement) and calls the KCI function with
#' automatically tuned kernel hyperparameters.
#'
#' @param x A numeric vector.
#' @param y A numeric vector of the same length as x.
#' @param z A matrix or dataframe of conditioning variables.
#' @param n_boot Integer. The number of bootstrap iterations. Default is 500.
#' @param sample_size Integer. The number of observations to sample per iteration.
#' @param show_progress Logical. Whether to show a progress bar. Default is TRUE.
#'
#' @return A numeric vector of bootstrap p-values from the KCI tests.
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' n <- 500
#' Z <- rnorm(n)
#' X <- Z + rnorm(n)
#' Y <- Z + rnorm(n)  # X _|_ Y | Z
#' p_vals <- bootstrappedKCI(X, Y, matrix(Z, ncol = 1), n_boot = 100)
#' }
#'
#' @importFrom utils setTxtProgressBar txtProgressBar
#' @export
bootstrappedKCI <- function(x, y, z, n_boot = 500, sample_size = length(x),
                             show_progress = TRUE) {
  if (!requireNamespace("CondIndTests", quietly = TRUE)) {
    stop("CondIndTests package is required. Install it with install.packages('CondIndTests')")
  }

  if (length(x) != length(y)) {
    stop("x and y must be of the same length")
  }

  if (nrow(as.matrix(z)) != length(x)) {
    stop("The number of rows in z must match the length of x and y")
  }

  n <- length(x)
  p_values <- numeric(n_boot)

  if (show_progress) {
    pb <- utils::txtProgressBar(min = 0, max = n_boot, style = 3)
  }

  for (i in 1:n_boot) {
    boot_idx <- sample(1:n, sample_size, replace = TRUE)

    boot_x <- x[boot_idx]
    boot_y <- y[boot_idx]
    boot_z <- as.matrix(z)[boot_idx, , drop = FALSE]

    kci_result <- tryCatch({
      CondIndTests::KCI(boot_x, boot_y, boot_z,
                        width = 0,
                        alpha = 0.05,
                        unbiased = FALSE,
                        gammaApprox = FALSE,
                        GP = TRUE,
                        nRepBs = 1000,
                        lambda = 0.001,
                        thresh = 1e-05,
                        numEig = length(boot_x),
                        verbose = FALSE)
    }, error = function(e) {
      return(list(pvalue = NA))
    })

    p_values[i] <- kci_result$pvalue

    if (show_progress) {
      utils::setTxtProgressBar(pb, i)
    }
  }

  if (show_progress) {
    close(pb)
  }

  return(p_values)
}


#' Bootstrapped Conditional Independence Test using bnlearn::ci.test
#'
#' Performs a bootstrapped conditional independence test using bnlearn's ci.test.
#' For each bootstrap iteration, it resamples the data and computes the p-value.
#'
#' @param x A numeric vector.
#' @param y A numeric vector of the same length as x.
#' @param z A matrix or dataframe of conditioning variables.
#' @param n_boot Integer. The number of bootstrap iterations. Default is 500.
#' @param sample_size Integer. The number of observations per iteration.
#' @param test Character. The conditional independence test to use. Default is "cor".
#' @param show_progress Logical. Whether to show a progress bar. Default is TRUE.
#'
#' @return A numeric vector of bootstrap p-values from ci.test.
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' n <- 500
#' x <- rnorm(n)
#' y <- rnorm(n)
#' z <- data.frame(Z1 = rnorm(n))
#' p_values <- bootstrappedCITest(x, y, z, n_boot = 100)
#' }
#'
#' @importFrom utils setTxtProgressBar txtProgressBar
#' @export
bootstrappedCITest <- function(x, y, z, n_boot = 500, sample_size = length(x),
                                test = "cor", show_progress = TRUE) {
  if (!requireNamespace("bnlearn", quietly = TRUE)) {
    stop("bnlearn package is required. Install it with install.packages('bnlearn')")
  }

  if (length(x) != length(y)) {
    stop("x and y must be of the same length")
  }

  if (nrow(as.matrix(z)) != length(x)) {
    stop("The number of rows in z must match the length of x and y")
  }

  n <- length(x)
  p_values <- numeric(n_boot)

  if (show_progress) {
    pb <- utils::txtProgressBar(min = 0, max = n_boot, style = 3)
  }

  for (i in 1:n_boot) {
    boot_idx <- sample(1:n, sample_size, replace = TRUE)

    boot_x <- x[boot_idx]
    boot_y <- y[boot_idx]
    boot_z <- as.data.frame(as.matrix(z)[boot_idx, , drop = FALSE])

    df <- data.frame(A = boot_x, B = boot_y, boot_z)
    cond_vars <- names(boot_z)

    test_result <- tryCatch({
      bnlearn::ci.test(x = "A", y = "B", z = cond_vars, data = df, test = test)
    }, error = function(e) {
      return(list(p.value = NA))
    })

    p_val <- test_result$p.value
    if (is.null(p_val) || length(p_val) == 0) p_val <- NA
    p_values[i] <- p_val

    if (show_progress) {
      utils::setTxtProgressBar(pb, i)
    }
  }

  if (show_progress) {
    close(pb)
  }

  return(p_values)
}


#' Bootstrapped Conditional Independence Test using GCM
#'
#' Performs a bootstrapped conditional independence test using gcm.test() from
#' the GeneralisedCovarianceMeasure package. This is a powerful nonparametric
#' test that uses generalized covariance measures.
#'
#' @param x A numeric vector.
#' @param y A numeric vector of the same length as x.
#' @param z A matrix or dataframe of conditioning variables.
#' @param n_boot Integer. The number of bootstrap iterations. Default is 500.
#' @param sample_size Integer. The number of observations per iteration.
#' @param verbose Logical. If TRUE, prints messages for errors. Default is FALSE.
#' @param show_progress Logical. Whether to show a progress bar. Default is TRUE.
#' @param ... Additional arguments passed to gcm.test().
#'
#' @return A numeric vector of bootstrap p-values from gcm.test.
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' n <- 500
#' Z <- rnorm(n)
#' X <- Z + rnorm(n)
#' Y <- Z + rnorm(n)
#' p_vals <- bootstrappedCondIndTest_GCM(X, Y, matrix(Z, ncol = 1),
#'                                        n_boot = 100, regr.method = 'gam')
#' }
#'
#' @importFrom utils setTxtProgressBar txtProgressBar
#' @export
bootstrappedCondIndTest_GCM <- function(x, y, z,
                                         n_boot = 500,
                                         sample_size = length(x),
                                         verbose = FALSE,
                                         show_progress = TRUE,
                                         ...) {
  if (!requireNamespace("GeneralisedCovarianceMeasure", quietly = TRUE)) {
    stop("GeneralisedCovarianceMeasure package is required.")
  }

  if (length(x) != length(y)) {
    stop("x and y must be of the same length")
  }

  if (nrow(as.matrix(z)) != length(x)) {
    stop("The number of rows in z must match the length of x and y")
  }

  n <- length(x)
  p_values <- numeric(n_boot)

  if (show_progress) {
    pb <- utils::txtProgressBar(min = 0, max = n_boot, style = 3)
  }

  for (i in 1:n_boot) {
    boot_idx <- sample(1:n, sample_size, replace = TRUE)

    boot_x <- x[boot_idx]
    boot_y <- y[boot_idx]
    boot_z <- as.data.frame(as.matrix(z)[boot_idx, , drop = FALSE])

    result <- tryCatch({
      GeneralisedCovarianceMeasure::gcm.test(boot_x, boot_y, boot_z, ...)
    }, error = function(e) {
      if (verbose) message("Error in gcm.test on iteration ", i, ": ", e$message)
      return(list(p.value = NA))
    })

    p_val <- result$p.value
    if (is.null(p_val) || length(p_val) == 0) p_val <- NA
    p_values[i] <- p_val

    if (show_progress) {
      utils::setTxtProgressBar(pb, i)
    }
  }

  if (show_progress) {
    close(pb)
  }

  return(p_values)
}


#' Bootstrapped Conditional Independence Test using CondIndTest
#'
#' Performs a bootstrapped conditional independence test using CondIndTest from
#' the CondIndTests package.
#'
#' @param x A numeric vector.
#' @param y A numeric vector of the same length as x.
#' @param z A matrix or dataframe of conditioning variables.
#' @param n_boot Integer. The number of bootstrap iterations. Default is 500.
#' @param sample_size Integer. The number of observations per iteration.
#' @param method Character. The method to use. Default is "KCI".
#' @param alpha Numeric. Significance level. Default is 0.05.
#' @param parsMethod List. Parameters for kernel hyperparameter selection.
#' @param verbose Logical. Passed to CondIndTest. Default is FALSE.
#' @param show_progress Logical. Whether to show a progress bar. Default is TRUE.
#'
#' @return A numeric vector of bootstrap p-values from CondIndTest.
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' n <- 500
#' Z <- rnorm(n)
#' X <- Z + rnorm(n)
#' Y <- Z + rnorm(n)
#' p_vals <- bootstrappedCondIndTest_CIT(X, Y, matrix(Z, ncol = 1), n_boot = 100)
#' }
#'
#' @importFrom utils setTxtProgressBar txtProgressBar
#' @export
bootstrappedCondIndTest_CIT <- function(x, y, z,
                                         n_boot = 500,
                                         sample_size = length(x),
                                         method = "KCI",
                                         alpha = 0.05,
                                         parsMethod = list(),
                                         verbose = FALSE,
                                         show_progress = TRUE) {
  if (!requireNamespace("CondIndTests", quietly = TRUE)) {
    stop("CondIndTests package is required.")
  }

  if (length(x) != length(y)) {
    stop("x and y must be of the same length")
  }

  if (nrow(as.matrix(z)) != length(x)) {
    stop("The number of rows in z must match the length of x and y")
  }

  n <- length(x)
  p_values <- numeric(n_boot)

  if (show_progress) {
    pb <- utils::txtProgressBar(min = 0, max = n_boot, style = 3)
  }

  for (i in 1:n_boot) {
    boot_idx <- sample(1:n, sample_size, replace = TRUE)
    boot_x <- x[boot_idx]
    boot_y <- y[boot_idx]
    boot_z <- as.data.frame(as.matrix(z)[boot_idx, , drop = FALSE])

    result <- tryCatch({
      CondIndTests::CondIndTest(boot_x, boot_y, boot_z,
                                method = method,
                                alpha = alpha,
                                parsMethod = parsMethod,
                                verbose = verbose)
    }, error = function(e) {
      message("Error in CondIndTest on iteration ", i, ": ", e$message)
      return(list(pvalue = NA))
    })

    p_val <- result$pvalue
    if (is.null(p_val) || length(p_val) == 0) p_val <- NA
    p_values[i] <- p_val

    if (show_progress) {
      utils::setTxtProgressBar(pb, i)
    }
  }

  if (show_progress) {
    close(pb)
  }

  return(p_values)
}
