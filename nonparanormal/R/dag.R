#' @title DAG Utility Functions
#' @description Functions for creating and manipulating DAG structures for use
#'   with copula-based models. These utilities help specify the conditioning
#'   structure when unconditioning conditional ranks.
#' @name dag
NULL

#' Create DAG Parent Structure from Edge List
#'
#' Converts an edge list representation of a DAG to a list of parent indices
#' for each variable.
#'
#' @param edges A matrix with 2 columns (from, to) where each row represents
#'   a directed edge from the first column to the second, OR a list of 2-element
#'   vectors c(from, to).
#' @param n_vars Total number of variables in the DAG.
#' @param var_names Optional character vector of variable names. If provided,
#'   edges can use names instead of indices.
#'
#' @return A list of length \code{n_vars} where element j contains the indices
#'   of the parents of variable j. Root nodes have \code{integer(0)}.
#'
#' @examples
#' # Simple chain: X1 -> X2 -> X3
#' edges <- matrix(c(1, 2, 2, 3), ncol = 2, byrow = TRUE)
#' parents <- dag_from_edges(edges, n_vars = 3)
#' # parents[[1]] = integer(0)  # X1 is root
#' # parents[[2]] = c(1)        # X2 has parent X1
#' # parents[[3]] = c(2)        # X3 has parent X2
#'
#' @export
dag_from_edges <- function(edges, n_vars, var_names = NULL) {
  # Initialize parent list
  parents <- vector("list", n_vars)
  for (j in seq_len(n_vars)) {
    parents[[j]] <- integer(0)
  }

  # Handle empty edge case
  if (is.null(edges) || (is.matrix(edges) && nrow(edges) == 0)) {
    return(parents)
  }

  # Convert list of edges to matrix
  if (is.list(edges) && !is.matrix(edges)) {
    edges <- do.call(rbind, edges)
  }

  # Convert to matrix if vector
  if (is.vector(edges) && length(edges) == 2) {
    edges <- matrix(edges, ncol = 2, byrow = TRUE)
  }

  # Ensure matrix
  edges <- as.matrix(edges)
  if (ncol(edges) != 2) {
    stop("edges must have exactly 2 columns (from, to)")
  }

  # Convert names to indices if needed
  if (!is.null(var_names) && is.character(edges[1, 1])) {
    edges_idx <- matrix(NA, nrow = nrow(edges), ncol = 2)
    for (i in seq_len(nrow(edges))) {
      edges_idx[i, 1] <- match(edges[i, 1], var_names)
      edges_idx[i, 2] <- match(edges[i, 2], var_names)
    }
    if (any(is.na(edges_idx))) {
      stop("Some edge endpoints not found in var_names")
    }
    edges <- edges_idx
  }

  # Ensure numeric
  edges <- matrix(as.integer(edges), ncol = 2)

  # Validate indices
  if (any(edges < 1) || any(edges > n_vars)) {
    stop("Edge indices must be between 1 and n_vars")
  }

  # Build parent lists
  for (i in seq_len(nrow(edges))) {
    from <- edges[i, 1]
    to <- edges[i, 2]
    parents[[to]] <- c(parents[[to]], from)
  }

  # Sort parents for consistency
  for (j in seq_len(n_vars)) {
    parents[[j]] <- sort(parents[[j]])
  }

  return(parents)
}


#' Create DAG Parent Structure from Adjacency Matrix
#'
#' Converts an adjacency matrix representation of a DAG to a list of parent
#' indices.
#'
#' @param adj_matrix A square matrix where \code{adj_matrix[i, j] = 1} indicates
#'   a directed edge from i to j (i is a parent of j).
#'
#' @return A list where element j contains the indices of the parents of
#'   variable j.
#'
#' @examples
#' # Chain: X1 -> X2 -> X3
#' adj <- matrix(0, 3, 3)
#' adj[1, 2] <- 1  # X1 -> X2
#' adj[2, 3] <- 1  # X2 -> X3
#' parents <- dag_from_adjacency(adj)
#'
#' @export
dag_from_adjacency <- function(adj_matrix) {
  if (!is.matrix(adj_matrix)) {
    stop("adj_matrix must be a matrix")
  }
  if (nrow(adj_matrix) != ncol(adj_matrix)) {
    stop("adj_matrix must be square")
  }

  n_vars <- nrow(adj_matrix)
  parents <- vector("list", n_vars)

  for (j in seq_len(n_vars)) {
    parents[[j]] <- which(adj_matrix[, j] != 0)
  }

  return(parents)
}


#' Create Simple Chain DAG
#'
#' Creates a DAG with a simple chain structure: X1 -> X2 -> X3 -> ... -> Xn.
#'
#' @param n_vars Number of variables in the chain.
#'
#' @return A list of parent indices representing the chain DAG.
#'
#' @examples
#' parents <- make_chain_dag(4)
#' # parents[[1]] = integer(0)
#' # parents[[2]] = c(1)
#' # parents[[3]] = c(2)
#' # parents[[4]] = c(3)
#'
#' @export
make_chain_dag <- function(n_vars) {
  if (n_vars < 1) {
    stop("n_vars must be at least 1")
  }

  parents <- vector("list", n_vars)
  parents[[1]] <- integer(0)

  if (n_vars > 1) {
    for (j in 2:n_vars) {
      parents[[j]] <- j - 1L
    }
  }

  return(parents)
}


#' Create Longitudinal DAG Structure
#'
#' Generates a DAG parent structure for longitudinal/temporal models with
#' multiple covariates per time point.
#'
#' The column ordering follows: Z1_1, Z2_1, ..., Zd_1, Z1_2, Z2_2, ..., Zd_2, ...
#' where d is the number of covariates and t indexes time.
#'
#' @param n_time Number of time points.
#' @param n_cov Number of covariates per time point.
#' @param structure Type of temporal structure:
#'   \describe{
#'     \item{"markov"}{(Default) Each covariate depends on same covariate at
#'       previous time (AR term) plus all preceding covariates at current time
#'       (cross-sectional). Z_d^t depends on Z_d^{t-1} and Z_1^t,...,Z_{d-1}^t.}
#'     \item{"ar1"}{Pure AR(1): Each Z_d^t depends only on Z_d^{t-1}.}
#'     \item{"full"}{D-vine structure: Each variable depends on ALL previous
#'       variables (equivalent to default behavior of uncondition_conditional_ranks).}
#'   }
#'
#' @return A list of parent indices with length \code{n_time * n_cov}.
#'   The list has an attribute "var_names" with column names.
#'
#' @details
#' For a model with T=3 time points and d=2 covariates:
#'
#' Column indices are:
#' \itemize{
#'   \item 1: Z1_1 (covariate 1, time 1)
#'   \item 2: Z2_1 (covariate 2, time 1)
#'   \item 3: Z1_2 (covariate 1, time 2)
#'   \item 4: Z2_2 (covariate 2, time 2)
#'   \item 5: Z1_3 (covariate 1, time 3)
#'   \item 6: Z2_3 (covariate 2, time 3)
#' }
#'
#' With "markov" structure:
#' \itemize{
#'   \item Z1_1: root
#'   \item Z2_1: depends on Z1_1 (cross-sectional)
#'   \item Z1_2: depends on Z1_1 (AR term, NOT Z2_1)
#'   \item Z2_2: depends on Z2_1 (AR term) and Z1_2 (cross-sectional)
#'   \item Z1_3: depends on Z1_2 (AR term, NOT Z2_2)
#'   \item Z2_3: depends on Z2_2 (AR term) and Z1_3 (cross-sectional)
#' }
#'
#' @examples
#' # 3 time points, 2 covariates, Markov structure
#' parents <- make_longitudinal_dag(n_time = 3, n_cov = 2, structure = "markov")
#' # parents[[1]] = integer(0)    # Z1_1: root
#' # parents[[2]] = c(1)          # Z2_1 | Z1_1
#' # parents[[3]] = c(1)          # Z1_2 | Z1_1 (NOT Z2_1!)
#' # parents[[4]] = c(2, 3)       # Z2_2 | Z2_1, Z1_2
#' # parents[[5]] = c(3)          # Z1_3 | Z1_2 (NOT Z2_2!)
#' # parents[[6]] = c(4, 5)       # Z2_3 | Z2_2, Z1_3
#'
#' @export
make_longitudinal_dag <- function(n_time, n_cov, structure = c("markov", "ar1", "full")) {
  structure <- match.arg(structure)

  if (n_time < 1) stop("n_time must be at least 1")
  if (n_cov < 1) stop("n_cov must be at least 1")

  n_vars <- n_time * n_cov
  parents <- vector("list", n_vars)

  # Helper function to convert (covariate d, time t) to column index
  # Using 1-based indexing
  col_idx <- function(d, t) {
    (t - 1) * n_cov + d
  }

  # Generate variable names
  var_names <- character(n_vars)
  for (t in seq_len(n_time)) {
    for (d in seq_len(n_cov)) {
      var_names[col_idx(d, t)] <- paste0("Z", d, "_", t)
    }
  }

  # Build parent structure based on specified structure type
  for (t in seq_len(n_time)) {
    for (d in seq_len(n_cov)) {
      j <- col_idx(d, t)

      if (structure == "full") {
        # D-vine: all previous columns
        if (j == 1) {
          parents[[j]] <- integer(0)
        } else {
          parents[[j]] <- seq_len(j - 1)
        }

      } else if (structure == "ar1") {
        # Pure AR(1): only same covariate at previous time
        if (t == 1) {
          parents[[j]] <- integer(0)
        } else {
          parents[[j]] <- col_idx(d, t - 1)
        }

      } else {
        # "markov" structure
        pa <- integer(0)

        # AR term: same covariate at previous time
        if (t > 1) {
          pa <- c(pa, col_idx(d, t - 1))
        }

        # Cross-sectional: all preceding covariates at current time
        if (d > 1) {
          pa <- c(pa, sapply(seq_len(d - 1), function(dd) col_idx(dd, t)))
        }

        parents[[j]] <- sort(pa)
      }
    }
  }

  attr(parents, "var_names") <- var_names
  return(parents)
}


#' Check if DAG Parent Structure is in Topological Order
#'
#' Validates that the parent structure is consistent with the column ordering
#' being a topological sort of the DAG. Specifically, all parents of variable j
#' must have index less than j.
#'
#' @param parents A list of parent indices as returned by DAG creation functions.
#'
#' @return \code{TRUE} if valid topological order, otherwise throws an error.
#'
#' @examples
#' parents <- list(integer(0), c(1), c(1, 2))  # Valid
#' check_dag_order(parents)  # Returns TRUE
#'
#' \dontrun{
#' parents_bad <- list(c(2), integer(0), c(1))  # Invalid: parent 2 > index 1
#' check_dag_order(parents_bad)  # Throws error
#' }
#'
#' @export
check_dag_order <- function(parents) {
  if (!is.list(parents)) {
    stop("parents must be a list")
  }

  n_vars <- length(parents)

  for (j in seq_len(n_vars)) {
    pa_j <- parents[[j]]
    if (length(pa_j) > 0) {
      if (any(pa_j >= j)) {
        bad_parents <- pa_j[pa_j >= j]
        stop(sprintf(
          "Invalid topological order: variable %d has parent(s) %s with index >= %d",
          j, paste(bad_parents, collapse = ", "), j
        ))
      }
      if (any(pa_j < 1)) {
        stop(sprintf("Invalid parent index for variable %d: indices must be >= 1", j))
      }
    }
  }

  return(TRUE)
}


#' Compute Topological Order of a DAG
#'
#' Given a parent structure, computes a valid topological ordering using
#' Kahn's algorithm. If the current ordering is already valid, returns
#' the identity permutation.
#'
#' @param parents A list of parent indices.
#'
#' @return An integer vector representing a valid topological order.
#'   Element i of the result is the index of the variable that should
#'   be in position i.
#'
#' @examples
#' # Already in topological order
#' parents <- list(integer(0), c(1), c(1, 2))
#' get_topological_order(parents)  # Returns c(1, 2, 3)
#'
#' @export
get_topological_order <- function(parents) {
  n_vars <- length(parents)

  # Check if already in topological order
  is_valid <- tryCatch({
    check_dag_order(parents)
    TRUE
  }, error = function(e) FALSE)

  if (is_valid) {
    return(seq_len(n_vars))
  }

  # Kahn's algorithm
  # Build adjacency list (children) and in-degree count
  children <- vector("list", n_vars)
  in_degree <- integer(n_vars)

  for (j in seq_len(n_vars)) {
    children[[j]] <- integer(0)
  }

  for (j in seq_len(n_vars)) {
    pa_j <- parents[[j]]
    in_degree[j] <- length(pa_j)
    for (p in pa_j) {
      children[[p]] <- c(children[[p]], j)
    }
  }

  # Initialize queue with root nodes
  queue <- which(in_degree == 0)
  order <- integer(0)

  while (length(queue) > 0) {
    # Take first element from queue
    node <- queue[1]
    queue <- queue[-1]
    order <- c(order, node)

    # Process children
    for (child in children[[node]]) {
      in_degree[child] <- in_degree[child] - 1
      if (in_degree[child] == 0) {
        queue <- c(queue, child)
      }
    }
  }

  if (length(order) != n_vars) {
    stop("DAG contains a cycle - cannot compute topological order")
  }

  return(order)
}


#' Print DAG Parent Structure
#'
#' Pretty-prints a DAG parent structure for debugging and inspection.
#'
#' @param parents A list of parent indices.
#' @param var_names Optional character vector of variable names. If not provided,
#'   uses the "var_names" attribute of parents (if present) or generates default
#'   names X1, X2, ...
#'
#' @return Invisibly returns the parents list.
#'
#' @examples
#' parents <- make_longitudinal_dag(2, 2, "markov")
#' print_dag(parents)
#'
#' @export
print_dag <- function(parents, var_names = NULL) {
  if (is.null(var_names)) {
    var_names <- attr(parents, "var_names")
  }
  if (is.null(var_names)) {
    var_names <- paste0("X", seq_along(parents))
  }

  cat("DAG Parent Structure:\n")
  cat(sprintf("  Variables: %d\n", length(parents)))
  cat("\n")

  for (j in seq_along(parents)) {
    pa_j <- parents[[j]]
    if (length(pa_j) == 0) {
      pa_str <- "(root)"
    } else {
      pa_str <- paste(var_names[pa_j], collapse = ", ")
    }
    cat(sprintf("  %s | %s\n", var_names[j], pa_str))
  }

  invisible(parents)
}
