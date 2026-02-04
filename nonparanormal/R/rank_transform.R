#' @title Rank Transformation Functions
#' @description Functions for transforming conditional ranks to marginal ranks
#'   using Gaussian copula translation.
#' @name rank_transform
NULL

#' Uncondition Conditional Ranks Using a Gaussian Copula Translation
#'
#' Given a matrix (or dataframe) of conditional ranks, this function iterates
#' over the variables and "unconditions" the conditional ranks to obtain their
#' marginal alternatives on the Gaussian (normal quantile) scale.
#'
#' By default, assumes a fully-connected D-vine structure where:
#' - Column 1 is unconditional (U_{Z1})
#' - Column 2 is U_{Z2|Z1}
#' - Column 3 is U_{Z3|Z1,Z2}, etc.
#'
#' For sparser conditioning structures (e.g., from a DAG/BN), use the
#' \code{parents} argument to specify which columns are the parents of each
#' variable. This is critical for preserving Markov properties in longitudinal
#' models where not all previous variables are parents.
#'
#' @param cond_ranks A matrix (or dataframe) of conditional ranks.
#' @param R A correlation matrix corresponding to the full Gaussian copula that
#'   models the joint distribution.
#' @param parents Optional list specifying the parent columns for each variable.
#'   \code{parents[[j]]} should be an integer vector of column indices that are
#'   parents of variable j (i.e., the conditioning set for the j-th conditional
#'   rank). If NULL (default), assumes fully-connected D-vine structure where
#'   each variable is conditioned on all previous variables. Use
#'   \code{\link{make_longitudinal_dag}} or \code{\link{make_chain_dag}} to
#'   generate appropriate parent structures.
#' @param check_order Logical. If TRUE (default), validates that parents are
#'   in topological order (all parents of variable j have index < j).
#'
#' @return A matrix of the same dimensions as \code{cond_ranks} containing the
#'   "unconditioned" values transformed back to the probability scale [0, 1].
#'
#' @details
#' The unconditioning formula for variable j given parents pa(j) is:
#' \deqn{X_j = \mu_{j|pa} + \sigma_{j|pa} \cdot \Phi^{-1}(U_{j|pa})}
#'
#' where:
#' \itemize{
#'   \item \eqn{\mu_{j|pa} = r' R_{pa}^{-1} X_{pa}} (conditional mean)
#'   \item \eqn{\sigma^2_{j|pa} = 1 - r' R_{pa}^{-1} r} (conditional variance)
#'   \item \eqn{r = R[j, pa(j)]} (correlations with parents)
#'   \item \eqn{R_{pa} = R[pa(j), pa(j)]} (parent correlation submatrix)
#' }
#'
#' Using the correct (sparse) parent set preserves Markov properties.
#' Using the D-vine structure (all previous variables) over-conditions
#' and may introduce spurious dependencies.
#'
#' @examples
#' \dontrun{
#' # Example 1: Fully-connected (D-vine, default behavior)
#' set.seed(123)
#' cond_ranks <- matrix(c(runif(5), runif(5), runif(5)), ncol = 3)
#' R <- matrix(c(1, 0.5, 0.3,
#'               0.5, 1, 0.4,
#'               0.3, 0.4, 1), nrow = 3, byrow = TRUE)
#' marginal_values <- uncondition_conditional_ranks(cond_ranks, R)
#'
#' # Example 2: DAG structure Z1 -> Z2 -> Z3 (Z3 only depends on Z2)
#' # parents[[1]] = integer(0)  # Z1 is root
#' # parents[[2]] = c(1)        # Z2 | Z1
#' # parents[[3]] = c(2)        # Z3 | Z2 (NOT Z1!)
#' parents <- list(integer(0), c(1), c(2))
#' marginal_values <- uncondition_conditional_ranks(cond_ranks, R, parents)
#'
#' # Example 3: Longitudinal model using helper function
#' parents <- make_longitudinal_dag(n_time = 3, n_cov = 2, structure = "markov")
#' marginal_ranks <- uncondition_conditional_ranks(cond_ranks_6col, R_6x6, parents)
#' }
#'
#' @seealso \code{\link{make_longitudinal_dag}}, \code{\link{make_chain_dag}},
#'   \code{\link{dag_from_edges}}, \code{\link{check_dag_order}}
#'
#' @importFrom stats qnorm pnorm
#' @export
uncondition_conditional_ranks <- function(cond_ranks, R, parents = NULL,
                                          check_order = TRUE) {
  cond_ranks <- as.matrix(cond_ranks)
  n <- nrow(cond_ranks)
  d <- ncol(cond_ranks)

  # If parents not specified, use default D-vine structure (all previous)
  if (is.null(parents)) {
    parents <- vector("list", d)
    parents[[1]] <- integer(0)
    if (d > 1) {
      for (j in 2:d) {
        parents[[j]] <- 1:(j - 1)
      }
    }
  }

  # Validate parents structure
  if (length(parents) != d) {
    stop("Length of 'parents' must equal number of columns in cond_ranks")
  }

  # Validate topological order if requested
  if (check_order) {
    for (j in seq_len(d)) {
      pa_j <- parents[[j]]
      if (length(pa_j) > 0) {
        if (any(pa_j >= j)) {
          bad_parents <- pa_j[pa_j >= j]
          stop(sprintf(
            "Invalid topological order: variable %d has parent(s) %s with index >= %d. ",
            j, paste(bad_parents, collapse = ", "), j
          ))
        }
        if (any(pa_j < 1)) {
          stop(sprintf("Invalid parent index for variable %d: indices must be >= 1", j))
        }
      }
    }
  }

  # Cache for R_sub inversions - avoid redundant solve() calls
  # Key: sorted parent indices as comma-separated string
  inversion_cache <- new.env(hash = TRUE, parent = emptyenv())

  # Helper function to get cached inverse
  get_R_sub_inv <- function(pa_idx) {
    if (length(pa_idx) == 0) return(NULL)
    pa_key <- paste(sort(pa_idx), collapse = ",")
    if (!exists(pa_key, envir = inversion_cache)) {
      R_sub <- R[pa_idx, pa_idx, drop = FALSE]
      inversion_cache[[pa_key]] <- solve(R_sub)
    }
    return(inversion_cache[[pa_key]])
  }

  # Pre-compute all necessary inversions (improves cache hits)
  for (j in seq_len(d)) {
    if (length(parents[[j]]) > 0) {
      get_R_sub_inv(parents[[j]])
    }
  }

  # Prepare a matrix to store the marginal (unconditioned) values
  marginal_values <- matrix(NA_real_, n, d)

  # Process each observation (each row)
  for (i in seq_len(n)) {
    x <- numeric(d)

    for (j in seq_len(d)) {
      pa_j <- parents[[j]]

      if (length(pa_j) == 0) {
        # Root variable: unconditional
        x[j] <- stats::qnorm(cond_ranks[i, j])
      } else {
        # Non-root: uncondition using the specified parents
        # Get correlation between j and its parents
        r_vec <- matrix(R[j, pa_j], nrow = 1)
        # Get cached inverse of parent correlation submatrix
        R_sub_inv <- get_R_sub_inv(pa_j)
        # Get already-computed Gaussian values for parents
        x_pa <- matrix(x[pa_j], ncol = 1)

        # Compute conditional mean: E[X_j | X_pa] = r' R_pa^{-1} x_pa
        mu_j <- as.numeric(r_vec %*% R_sub_inv %*% x_pa)

        # Compute conditional variance: Var(X_j | X_pa) = 1 - r' R_pa^{-1} r
        sigma2_j <- 1 - as.numeric(r_vec %*% R_sub_inv %*% t(r_vec))
        sigma_j <- sqrt(max(sigma2_j, 1e-10))  # Numerical stability

        # "Uncondition": X_j = mu_j + sigma_j * Phi^{-1}(U_j|pa)
        x[j] <- mu_j + sigma_j * stats::qnorm(cond_ranks[i, j])
      }
    }
    marginal_values[i, ] <- x
  }

  return(stats::pnorm(marginal_values))
}
