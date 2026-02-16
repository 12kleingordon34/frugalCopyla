#' @title Gaussian Copula DAG Estimator
#' @description Fit a Gaussian copula with DAG constraints using a linear-Gaussian
#'   Bayesian network formulation. DAG constraints are encoded by sparsity in
#'   the coefficient matrix B and independent innovations (diagonal D).
#' @name gaussian_copula_dag
NULL

#' Fit Gaussian Copula with DAG Structure
#'
#' Projects onto DAG-constrained Gaussian BN family. Uses linear-Gaussian BN:
#' Z_j = sum_{k in Pa(j)} beta_{kj} * Z_k + epsilon_j, where epsilon_j ~ N(0, sigma^2_j)
#' are independent innovations.
#'
#' The implied covariance is: Sigma = T^{-1} D T^{-T} where T = I - B'.
#'
#' @param U Matrix of pseudo-observations (n x d) in (0, 1)
#' @param parents List mapping variable index or name -> parent indices or names.
#'   Can use either integer indices (1-based) or character variable names.
#'   E.g., list(integer(0), c(1), c(1, 2)) or list(Z1 = character(0), Z2 = "Z1", Z3 = c("Z1", "Z2"))
#' @param topo_order Optional integer vector specifying topological order. If NULL,
#'   computed via Kahn's algorithm from parents.
#' @param centre Logical: centre Z columns (default TRUE). Required for no-intercept
#'   regression. If FALSE, stop with error (not supported without intercept).
#' @param eps Floor for sigma^2_j to ensure positive definiteness (default: 1e-10)
#'
#' @return List with:
#'   \describe{
#'     \item{B}{Coefficient matrix (d x d, sparse by DAG): B[k,j] = beta_{kj} (edge k->j)}
#'     \item{sigma2}{Vector of innovation variances (length d)}
#'     \item{T_mat}{I - t(B), unit lower triangular under topological order}
#'     \item{D}{Diagonal matrix of innovation variances}
#'     \item{Sigma}{Implied covariance matrix}
#'     \item{R}{Correlation matrix (for Gaussian copula)}
#'     \item{L}{Cholesky factor of R (lower triangular: R = L %*% t(L))}
#'     \item{regressions}{Per-node regression summaries}
#'     \item{topo_order}{Topological order used}
#'     \item{var_names}{Variable names (for consistent ordering)}
#'     \item{parents}{The DAG parent structure used (normalised to indices)}
#'     \item{d}{Number of variables}
#'   }
#'
#' @details
#' Given a prespecified DAG, we estimate a Gaussian-copula dependence model
#' whose latent Gaussian variables satisfy a linear-Gaussian BN consistent
#' with that DAG. DAG constraints are encoded by:
#' 1. Sparsity in B: B[k,j] = 0 for k not in Pa(j)
#' 2. Independent innovations: D is diagonal
#'
#' Parameters are estimated via OLS for each node conditional on its parents
#' in topological order.
#'
#' **Critical**: Zeros in B encode the DAG, NOT zeros in L or in the precision
#' matrix Omega = Sigma^{-1}.
#'
#' @examples
#' \dontrun{
#' # Simple chain: X1 -> X2 -> X3
#' n <- 500
#' set.seed(123)
#' U <- matrix(runif(n * 3), ncol = 3)
#' colnames(U) <- c("X1", "X2", "X3")
#'
#' parents <- list(
#'   X1 = character(0),
#'   X2 = "X1",
#'   X3 = "X2"
#' )
#'
#' fit <- fit_gaussian_copula_dag(U, parents)
#'
#' # B should have non-zero entries only at [1,2] and [2,3]
#' print(fit$B)
#' }
#'
#' @importFrom stats qnorm lm residuals var cov2cor
#' @export
fit_gaussian_copula_dag <- function(U, parents, topo_order = NULL,
                                     centre = TRUE, eps = 1e-10) {
  # Validate inputs
  U <- as.matrix(U)
  n <- nrow(U)
  d <- ncol(U)

  # Get variable names
  var_names <- colnames(U)
  if (is.null(var_names)) {
    var_names <- paste0("V", seq_len(d))
    colnames(U) <- var_names
  }

  # Normalise parents to integer indices
  parents_idx <- normalise_parents(parents, var_names)

  # Validate parents length
  if (length(parents_idx) != d) {
    stop(sprintf("Length of parents (%d) must equal number of columns (%d)",
                 length(parents_idx), d))
  }

  # Compute topological order if not provided
  if (is.null(topo_order)) {
    topo_order <- topological_sort_kahn(parents_idx)
  } else {
    # Validate provided order
    if (length(topo_order) != d || !setequal(topo_order, seq_len(d))) {
      stop("topo_order must be a permutation of 1:d")
    }
    # Check it's a valid topological order
    validate_topological_order(topo_order, parents_idx)
  }

  # Transform to latent Gaussian
  Z <- qnorm(U)

  # Centre if requested (required for no-intercept regression)
  if (centre) {
    Z <- scale(Z, center = TRUE, scale = FALSE)
  } else {
    stop("centre = FALSE not supported: no-intercept regression requires centred data")
  }

  # Initialise coefficient matrix and variance vector
  B <- matrix(0, d, d)
  rownames(B) <- colnames(B) <- var_names
  sigma2 <- numeric(d)
  names(sigma2) <- var_names
  regressions <- vector("list", d)
  names(regressions) <- var_names

  # Fit OLS for each node in topological order
  for (j in topo_order) {
    pa_j <- parents_idx[[j]]
    var_name <- var_names[j]

    if (length(pa_j) == 0) {
      # Root node: no parents
      var_Z_j <- var(Z[, j])

      if (var_Z_j < eps) {
        warning(sprintf("Near-constant column %d (%s): variance = %.2e",
                        j, var_name, var_Z_j))
      }

      sigma2[j] <- max(var_Z_j, eps)
      regressions[[j]] <- list(
        node = var_name,
        parents = character(0),
        coefficients = NULL,
        residual_var = sigma2[j],
        is_root = TRUE
      )
    } else {
      # Non-root: OLS regression Z_j ~ Z_{Pa(j)} - 1 (no intercept, Z is centred)
      X_design <- Z[, pa_j, drop = FALSE]
      y <- Z[, j]

      # Check rank of design matrix
      qr_X <- qr(X_design)
      if (qr_X$rank < length(pa_j)) {
        stop(sprintf(
          "Rank-deficient design matrix for node %d (%s): rank = %d, expected %d. Parents: %s",
          j, var_name, qr_X$rank, length(pa_j),
          paste(var_names[pa_j], collapse = ", ")
        ))
      }

      # Fit OLS without intercept
      fit_lm <- lm(y ~ X_design - 1)
      beta_j <- coef(fit_lm)
      resid_j <- residuals(fit_lm)
      resid_var <- var(resid_j)

      # Store coefficients in B
      B[pa_j, j] <- beta_j

      # Store variance (with floor)
      sigma2[j] <- max(resid_var, eps)

      regressions[[j]] <- list(
        node = var_name,
        parents = var_names[pa_j],
        coefficients = setNames(beta_j, var_names[pa_j]),
        residual_var = sigma2[j],
        is_root = FALSE
      )
    }
  }

  # Build T = I - B'
  T_mat <- diag(d) - t(B)
  rownames(T_mat) <- colnames(T_mat) <- var_names

  # Build D = diag(sigma2)
  D <- diag(sigma2, nrow = d)
  rownames(D) <- colnames(D) <- var_names

  # Compute implied covariance: Sigma = T^{-1} D T^{-T}
  T_inv <- solve(T_mat)
  Sigma <- T_inv %*% D %*% t(T_inv)

  # Ensure numerical symmetry
  Sigma <- (Sigma + t(Sigma)) / 2
  rownames(Sigma) <- colnames(Sigma) <- var_names

  # Compute correlation matrix
  R <- cov2cor(Sigma)

  # Compute Cholesky factor (lower triangular: R = L %*% t(L))
  L <- tryCatch({
    t(chol(R))
  }, error = function(e) {
    warning("chol(R) failed; adding small jitter to diagonal")
    t(chol(R + diag(eps, d)))
  })
  rownames(L) <- colnames(L) <- var_names

  return(list(
    B = B,
    sigma2 = sigma2,
    T_mat = T_mat,
    D = D,
    Sigma = Sigma,
    R = R,
    L = L,
    regressions = regressions,
    topo_order = topo_order,
    var_names = var_names,
    parents = parents_idx,
    d = d
  ))
}


#' Convert Parents List to Validated Integer Indices
#'
#' Accepts either character names or integer indices; validates all exist.
#'
#' @param parents List of parent vectors (names or indices)
#' @param var_names Character vector of variable names
#'
#' @return List of integer index vectors
#'
#' @keywords internal
normalise_parents <- function(parents, var_names) {
  d <- length(var_names)

  # Handle named vs unnamed list
  if (!is.null(names(parents))) {
    # Reorder to match var_names
    if (!setequal(names(parents), var_names)) {
      missing <- setdiff(var_names, names(parents))
      extra <- setdiff(names(parents), var_names)
      msg <- ""
      if (length(missing) > 0) {
        msg <- paste0("Missing parents for: ", paste(missing, collapse = ", "))
      }
      if (length(extra) > 0) {
        if (nchar(msg) > 0) msg <- paste0(msg, "; ")
        msg <- paste0(msg, "Unknown variables: ", paste(extra, collapse = ", "))
      }
      stop(msg)
    }
    parents <- parents[var_names]
  } else {
    if (length(parents) != d) {
      stop(sprintf("Length of parents (%d) must equal number of variables (%d)",
                   length(parents), d))
    }
  }

  # Convert each element to integer indices
  parents_idx <- lapply(seq_along(parents), function(j) {
    pa <- parents[[j]]

    if (length(pa) == 0 || (length(pa) == 1 && (is.na(pa) || pa == ""))) {
      return(integer(0))
    }

    if (is.character(pa)) {
      idx <- match(pa, var_names)
      if (any(is.na(idx))) {
        bad <- pa[is.na(idx)]
        stop(sprintf("Unknown parent name(s) '%s' for node %d (%s)",
                     paste(bad, collapse = "', '"), j, var_names[j]))
      }
      return(idx)
    } else if (is.numeric(pa)) {
      pa <- as.integer(pa)
      if (any(pa < 1 | pa > d)) {
        stop(sprintf("Parent index out of bounds for node %d (%s)", j, var_names[j]))
      }
      return(pa)
    } else {
      stop("parents must be character names or integer indices")
    }
  })

  names(parents_idx) <- var_names
  return(parents_idx)
}


#' Topological Sort Using Kahn's Algorithm
#'
#' Computes a topological ordering of nodes given parent structure.
#' Also performs cycle detection.
#'
#' @param parents List of length d, where parents[[j]] = indices of parents of j
#'
#' @return Integer vector of topological order
#'
#' @keywords internal
topological_sort_kahn <- function(parents) {
  d <- length(parents)

  # Precompute children adjacency list
  children <- vector("list", d)
  for (j in seq_len(d)) {
    children[[j]] <- integer(0)
  }
  for (j in seq_len(d)) {
    for (pa in parents[[j]]) {
      children[[pa]] <- c(children[[pa]], j)
    }
  }

  # Compute in-degrees
  in_degree <- lengths(parents)

  # Kahn's algorithm
  queue <- which(in_degree == 0)
  order <- integer(0)

  while (length(queue) > 0) {
    node <- queue[1]
    queue <- queue[-1]
    order <- c(order, node)

    # Update children's in-degrees
    for (child in children[[node]]) {
      in_degree[child] <- in_degree[child] - 1
      if (in_degree[child] == 0) {
        queue <- c(queue, child)
      }
    }
  }

  if (length(order) != d) {
    stop("Cycle detected in DAG: cannot compute topological order")
  }

  return(order)
}


#' Validate Topological Order
#'
#' Checks that a given order is a valid topological order for the DAG.
#'
#' @param topo_order Integer vector of topological order
#' @param parents List of parent indices
#'
#' @return NULL (invisible), or stops with error
#'
#' @keywords internal
validate_topological_order <- function(topo_order, parents) {
  d <- length(parents)
  pos <- integer(d)
  pos[topo_order] <- seq_len(d)

  for (j in seq_len(d)) {
    for (pa in parents[[j]]) {
      if (pos[pa] >= pos[j]) {
        stop(sprintf(
          "Invalid topological order: parent %d appears at position %d, but child %d at position %d",
          pa, pos[pa], j, pos[j]
        ))
      }
    }
  }

  invisible(NULL)
}


#' Fit Reference Gaussian BN from Raw Covariate Data
#'
#' Convenience wrapper that takes raw covariate values (or generates them via
#' \code{generate_fn}), computes empirical ranks, and fits a DAG-constrained
#' Gaussian copula via \code{\link{fit_gaussian_copula_dag}}.
#'
#' This encapsulates the one-time calibration step needed for Route B
#' (SEM-propagation of conditional PITs). The returned object contains the
#' coefficient matrix \eqn{B}, innovation variances \eqn{\sigma^2}, implied
#' covariance \eqn{\Sigma}, and correlation matrix \eqn{R} that are passed
#' to \code{\link{uncondition_conditional_ranks}}.
#'
#' @param Z_ref Matrix of observed covariate values (n x d). Each column is a
#'   covariate. Ignored if \code{generate_fn} is provided and \code{n_ref} is
#'   given.
#' @param parents List mapping variable index or name to parent indices/names.
#'   Same format as in \code{\link{fit_gaussian_copula_dag}}.
#' @param n_ref Integer. If \code{generate_fn} is provided, number of reference
#'   samples to generate. Default \code{NULL} (use \code{Z_ref} directly).
#' @param generate_fn Optional function that generates a reference dataset.
#'   Should return a matrix with \code{n_ref} rows and \code{d} columns.
#'   Called as \code{generate_fn(n_ref)}.
#' @param ... Additional arguments passed to \code{\link{fit_gaussian_copula_dag}}.
#'
#' @return The fitted model list from \code{\link{fit_gaussian_copula_dag}},
#'   containing B, sigma2, Sigma, R, L, parents, etc.
#'
#' @examples
#' \dontrun{
#' # From raw data
#' Z <- cbind(rgamma(1000, 2, 1), rgamma(1000, 3, 1))
#' parents <- list(integer(0), c(1))
#' fit <- fit_reference_gaussian_bn(Z, parents)
#'
#' # From a generator function
#' gen_fn <- function(n) cbind(rgamma(n, 2, 1), rgamma(n, 3, 1))
#' fit <- fit_reference_gaussian_bn(NULL, parents, n_ref = 10000,
#'                                   generate_fn = gen_fn)
#' }
#'
#' @export
fit_reference_gaussian_bn <- function(Z_ref, parents, n_ref = NULL,
                                       generate_fn = NULL, ...) {
  # Generate reference data if a generator function is provided
  if (!is.null(generate_fn) && !is.null(n_ref)) {
    Z_ref <- generate_fn(n_ref)
  }

  if (is.null(Z_ref)) {
    stop("Either Z_ref must be provided or both generate_fn and n_ref must be specified")
  }

  Z_ref <- as.matrix(Z_ref)
  n <- nrow(Z_ref)
  d <- ncol(Z_ref)

  # Compute empirical ranks: rank(z) / (n + 1) to avoid 0 and 1
  U_ref <- apply(Z_ref, 2, function(col) {
    rank(col, ties.method = "average") / (n + 1)
  })

  # Preserve column names if present
  if (!is.null(colnames(Z_ref))) {
    colnames(U_ref) <- colnames(Z_ref)
  }

  # Fit DAG-constrained Gaussian copula
  fit <- fit_gaussian_copula_dag(U_ref, parents, ...)

  return(fit)
}
