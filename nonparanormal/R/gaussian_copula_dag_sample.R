#' @title Sampling from Gaussian Copula DAG
#' @description Functions for sampling from a fitted Gaussian copula DAG model.
#' @name gaussian_copula_dag_sample
NULL

#' Sample from Fitted Gaussian Copula DAG
#'
#' Generates samples from a Gaussian copula model fitted with DAG constraints.
#'
#' @param n Number of samples to generate
#' @param fit Output from \code{\link{fit_gaussian_copula_dag}}
#' @param seed Random seed for reproducibility (optional)
#' @param eps Floor/ceiling for U to avoid extreme qnorm() issues (default: 1e-10)
#'
#' @return Matrix U of pseudo-observations (n x d) with colnames = fit$var_names.
#'   Values are in (eps, 1 - eps).
#'
#' @details
#' Sampling procedure:
#' 1. Generate E ~ N(0, I_d) (n x d matrix of iid standard normals)
#' 2. Transform: Z = E %*% t(L) where L is lower-triangular Cholesky of R
#' 3. Convert to uniform: U = Phi(Z)
#' 4. Clamp to (eps, 1 - eps)
#'
#' The resulting samples have marginal Uniform(0,1) distributions and
#' dependence structure matching the fitted Gaussian copula correlation R.
#'
#' @examples
#' \dontrun{
#' # Fit a DAG copula
#' n_fit <- 500
#' U_data <- matrix(runif(n_fit * 3), ncol = 3)
#' colnames(U_data) <- c("X1", "X2", "X3")
#' parents <- list(X1 = character(0), X2 = "X1", X3 = "X2")
#' fit <- fit_gaussian_copula_dag(U_data, parents)
#'
#' # Sample from fitted model
#' U_new <- sample_gaussian_copula_dag(n = 1000, fit, seed = 456)
#'
#' # Verify correlation recovery
#' R_emp <- cor(qnorm(U_new))
#' print(max(abs(R_emp - fit$R)))  # Should be small
#' }
#'
#' @importFrom stats rnorm pnorm
#' @export
sample_gaussian_copula_dag <- function(n, fit, seed = NULL, eps = 1e-10) {
  # Validate fit object
  required_fields <- c("L", "R", "d", "var_names")
  missing_fields <- setdiff(required_fields, names(fit))
  if (length(missing_fields) > 0) {
    stop("Invalid fit object: missing fields: ", paste(missing_fields, collapse = ", "))
  }

  d <- fit$d

  # Set seed if provided
  if (!is.null(seed)) {
    set.seed(seed)
  }

  # Generate standard normal samples: E ~ N(0, I_d)
  E <- matrix(rnorm(n * d), nrow = n, ncol = d)

  # Transform via Cholesky: Z = E %*% t(L)
  # L is lower-triangular: R = L %*% t(L)
  # So Z has covariance L %*% t(L) = R
  Z <- E %*% t(fit$L)

  # Convert to uniform: U = Phi(Z)
  U <- pnorm(Z)

  # Clamp to (eps, 1 - eps) to avoid extreme values
  U <- pmax(pmin(U, 1 - eps), eps)

  # Preserve column ordering and names
  colnames(U) <- fit$var_names

  return(U)
}


#' Sample Conditional Ranks from Gaussian Copula DAG
#'
#' Generates samples in conditional rank form, following the DAG structure.
#' This is useful when you need to transform through conditional CDFs
#' (e.g., for generating data with parametric conditional distributions).
#'
#' @param n Number of samples to generate
#' @param fit Output from \code{\link{fit_gaussian_copula_dag}}
#' @param seed Random seed for reproducibility (optional)
#' @param eps Floor/ceiling for U to avoid extreme values (default: 1e-10)
#'
#' @return List with:
#'   \describe{
#'     \item{U_marginal}{Matrix of marginal pseudo-observations (n x d)}
#'     \item{U_conditional}{Matrix of conditional pseudo-observations (n x d).
#'       Column j contains U_{j|Pa(j)}, i.e., the conditional rank of variable j
#'       given its parents.}
#'     \item{Z}{Matrix of latent Gaussian values (n x d)}
#'   }
#'
#' @details
#' For each node j with parents Pa(j):
#' - Z_j | Z_{Pa(j)} ~ N(mu_{j|Pa}, sigma^2_{j|Pa})
#' - U_{j|Pa(j)} = Phi((Z_j - mu_{j|Pa}) / sigma_{j|Pa})
#'
#' This is the inverse of the "unconditioning" operation in vine sampling.
#'
#' @examples
#' \dontrun{
#' fit <- fit_gaussian_copula_dag(U_data, parents)
#' samples <- sample_conditional_gaussian_copula_dag(1000, fit, seed = 789)
#'
#' # U_conditional[, j] is uniform given parents
#' # This can be fed through F^{-1}_{j|Pa(j)} to get data
#' }
#'
#' @importFrom stats rnorm pnorm
#' @export
sample_conditional_gaussian_copula_dag <- function(n, fit, seed = NULL, eps = 1e-10) {
  # Validate fit object
  required_fields <- c("L", "R", "d", "var_names", "parents", "topo_order")
  missing_fields <- setdiff(required_fields, names(fit))
  if (length(missing_fields) > 0) {
    stop("Invalid fit object: missing fields: ", paste(missing_fields, collapse = ", "))
  }

  d <- fit$d
  R <- fit$R
  parents <- fit$parents
  var_names <- fit$var_names

  # Set seed if provided
  if (!is.null(seed)) {
    set.seed(seed)
  }

  # Generate standard normal samples and transform to correlated Gaussians
  E <- matrix(rnorm(n * d), nrow = n, ncol = d)
  Z <- E %*% t(fit$L)
  colnames(Z) <- var_names

  # Marginal pseudo-observations
  U_marginal <- pnorm(Z)
  U_marginal <- pmax(pmin(U_marginal, 1 - eps), eps)
  colnames(U_marginal) <- var_names

  # Conditional pseudo-observations
  U_conditional <- matrix(NA_real_, nrow = n, ncol = d)
  colnames(U_conditional) <- var_names

  # Cache for R_sub inversions
  inversion_cache <- new.env(hash = TRUE, parent = emptyenv())

  get_R_sub_inv <- function(pa_idx) {
    if (length(pa_idx) == 0) return(NULL)
    pa_key <- paste(sort(pa_idx), collapse = ",")
    if (!exists(pa_key, envir = inversion_cache)) {
      R_sub <- R[pa_idx, pa_idx, drop = FALSE]
      inversion_cache[[pa_key]] <- solve(R_sub)
    }
    return(inversion_cache[[pa_key]])
  }

  # Compute conditional ranks for each variable
  for (j in seq_len(d)) {
    pa_j <- parents[[j]]

    if (length(pa_j) == 0) {
      # Root node: conditional = marginal
      U_conditional[, j] <- U_marginal[, j]
    } else {
      # Non-root: compute conditional
      # r = R[j, Pa(j)]
      r_vec <- R[j, pa_j, drop = FALSE]
      R_sub_inv <- get_R_sub_inv(pa_j)

      # For each observation
      for (i in seq_len(n)) {
        z_pa <- Z[i, pa_j]
        z_j <- Z[i, j]

        # Conditional mean: mu_{j|Pa} = r' R_{Pa}^{-1} z_Pa
        mu_j <- as.numeric(r_vec %*% R_sub_inv %*% z_pa)

        # Conditional variance: sigma^2_{j|Pa} = 1 - r' R_{Pa}^{-1} r
        sigma2_j <- 1 - as.numeric(r_vec %*% R_sub_inv %*% t(r_vec))
        sigma_j <- sqrt(max(sigma2_j, 1e-10))

        # Conditional standardised value
        z_j_cond <- (z_j - mu_j) / sigma_j

        # Convert to uniform
        U_conditional[i, j] <- pnorm(z_j_cond)
      }
    }
  }

  # Clamp conditional ranks
  U_conditional <- pmax(pmin(U_conditional, 1 - eps), eps)

  return(list(
    U_marginal = U_marginal,
    U_conditional = U_conditional,
    Z = Z
  ))
}


#' Simulate from a Gaussian BN SEM
#'
#' Generates samples by walking the DAG in topological order using the
#' structural equation model (SEM):
#' \deqn{Z_j = \sum_{k \in Pa(j)} B_{kj} Z_k + \epsilon_j, \quad \epsilon_j \sim N(0, \sigma^2_j)}
#'
#' Unlike \code{\link{sample_gaussian_copula_dag}} (which uses Cholesky),
#' this function follows the DAG recursion explicitly, which is conceptually
#' appropriate for the GAUSS baseline simulator where Markov properties
#' hold exactly by construction.
#'
#' @param n Number of samples to generate
#' @param fit Output from \code{\link{fit_gaussian_copula_dag}} or
#'   \code{\link{fit_reference_gaussian_bn}}
#' @param seed Random seed for reproducibility (optional)
#' @param eps Floor/ceiling for U to avoid extreme qnorm() issues (default: 1e-10)
#'
#' @return List with:
#'   \describe{
#'     \item{Q_tilde_Z}{Matrix of latent Gaussian scores (n x d)}
#'     \item{tilde_U}{Matrix of marginal pseudo-observations (n x d) in (eps, 1 - eps)}
#'   }
#'
#' @importFrom stats rnorm pnorm
#' @export
simulate_gaussian_bn_sem <- function(n, fit, seed = NULL, eps = 1e-10) {
  if (!is.null(seed)) set.seed(seed)

  d <- fit$d
  B <- fit$B
  sigma2 <- fit$sigma2
  topo_order <- fit$topo_order
  parents <- fit$parents

  Q <- matrix(0, nrow = n, ncol = d)
  colnames(Q) <- fit$var_names

  for (j in topo_order) {
    eps_j <- rnorm(n, mean = 0, sd = sqrt(sigma2[j]))
    pa_j <- parents[[j]]
    if (length(pa_j) == 0) {
      Q[, j] <- eps_j
    } else {
      Q[, j] <- Q[, pa_j, drop = FALSE] %*% B[pa_j, j] + eps_j
    }
  }

  tilde_U <- pnorm(Q)
  tilde_U <- pmax(pmin(tilde_U, 1 - eps), eps)
  colnames(tilde_U) <- fit$var_names

  list(Q_tilde_Z = Q, tilde_U = tilde_U)
}
