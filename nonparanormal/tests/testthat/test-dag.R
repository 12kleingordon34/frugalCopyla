# Test suite for DAG utility functions

test_that("dag_from_edges creates correct parent structure for chain", {
  # Chain: X1 -> X2 -> X3
  edges <- matrix(c(1, 2, 2, 3), ncol = 2, byrow = TRUE)
  parents <- dag_from_edges(edges, n_vars = 3)

  expect_equal(parents[[1]], integer(0))
  expect_equal(parents[[2]], c(1))
  expect_equal(parents[[3]], c(2))
})

test_that("dag_from_edges handles empty edges", {
  parents <- dag_from_edges(NULL, n_vars = 3)

  expect_equal(parents[[1]], integer(0))
  expect_equal(parents[[2]], integer(0))
  expect_equal(parents[[3]], integer(0))
})

test_that("dag_from_edges handles multiple parents", {
  # X1, X2 -> X3
  edges <- matrix(c(1, 3, 2, 3), ncol = 2, byrow = TRUE)
  parents <- dag_from_edges(edges, n_vars = 3)

  expect_equal(parents[[1]], integer(0))
  expect_equal(parents[[2]], integer(0))
  expect_equal(sort(parents[[3]]), c(1, 2))
})

test_that("dag_from_edges handles named edges", {
  edges <- matrix(c("A", "B", "B", "C"), ncol = 2, byrow = TRUE)
  parents <- dag_from_edges(edges, n_vars = 3, var_names = c("A", "B", "C"))

  expect_equal(parents[[1]], integer(0))
  expect_equal(parents[[2]], c(1))
  expect_equal(parents[[3]], c(2))
})

test_that("dag_from_adjacency works correctly", {
  # Chain: X1 -> X2 -> X3
  adj <- matrix(0, 3, 3)
  adj[1, 2] <- 1
  adj[2, 3] <- 1
  parents <- dag_from_adjacency(adj)

  expect_equal(parents[[1]], integer(0))
  expect_equal(parents[[2]], c(1))
  expect_equal(parents[[3]], c(2))
})

test_that("make_chain_dag creates correct structure", {
  parents <- make_chain_dag(4)

  expect_equal(length(parents), 4)
  expect_equal(parents[[1]], integer(0))
  expect_equal(parents[[2]], 1L)
  expect_equal(parents[[3]], 2L)
  expect_equal(parents[[4]], 3L)
})

test_that("make_chain_dag handles single variable", {
  parents <- make_chain_dag(1)

  expect_equal(length(parents), 1)
  expect_equal(parents[[1]], integer(0))
})

test_that("make_longitudinal_dag creates correct markov structure", {
  # 3 time points, 2 covariates
  parents <- make_longitudinal_dag(n_time = 3, n_cov = 2, structure = "markov")

  # Column order: Z1_1 (1), Z2_1 (2), Z1_2 (3), Z2_2 (4), Z1_3 (5), Z2_3 (6)
  expect_equal(length(parents), 6)

  # Z1_1: root
  expect_equal(parents[[1]], integer(0))

  # Z2_1 | Z1_1
  expect_equal(parents[[2]], c(1))

  # Z1_2 | Z1_1 (NOT Z2_1 - this is the key!)
  expect_equal(parents[[3]], c(1))

  # Z2_2 | Z2_1, Z1_2
  expect_equal(sort(parents[[4]]), c(2, 3))

  # Z1_3 | Z1_2 (NOT Z2_2!)
  expect_equal(parents[[5]], c(3))

  # Z2_3 | Z2_2, Z1_3
  expect_equal(sort(parents[[6]]), c(4, 5))
})

test_that("make_longitudinal_dag markov structure has correct var_names", {
  parents <- make_longitudinal_dag(n_time = 2, n_cov = 2, structure = "markov")
  var_names <- attr(parents, "var_names")

  expect_equal(var_names, c("Z1_1", "Z2_1", "Z1_2", "Z2_2"))
})

test_that("make_longitudinal_dag ar1 structure works", {
  # Pure AR(1): each Z_d^t depends only on Z_d^{t-1}
  parents <- make_longitudinal_dag(n_time = 3, n_cov = 2, structure = "ar1")

  # Z1_1: root
  expect_equal(parents[[1]], integer(0))
  # Z2_1: root (no same-covariate lag at t=1)
  expect_equal(parents[[2]], integer(0))
  # Z1_2 | Z1_1
  expect_equal(parents[[3]], c(1))
  # Z2_2 | Z2_1
  expect_equal(parents[[4]], c(2))
  # Z1_3 | Z1_2
  expect_equal(parents[[5]], c(3))
  # Z2_3 | Z2_2
  expect_equal(parents[[6]], c(4))
})

test_that("make_longitudinal_dag full structure equals D-vine", {
  parents <- make_longitudinal_dag(n_time = 2, n_cov = 2, structure = "full")

  # D-vine: each variable depends on all previous
  expect_equal(parents[[1]], integer(0))
  expect_equal(parents[[2]], c(1))
  expect_equal(parents[[3]], c(1, 2))
  expect_equal(parents[[4]], c(1, 2, 3))
})

test_that("check_dag_order accepts valid order", {
  parents <- list(integer(0), c(1), c(1, 2))

  expect_true(check_dag_order(parents))
})

test_that("check_dag_order rejects invalid order", {
  # Invalid: variable 1 has parent 2 (which comes later)
  parents_bad <- list(c(2), integer(0), c(1))

  expect_error(check_dag_order(parents_bad), "Invalid topological order")
})

test_that("check_dag_order rejects parent equal to self", {
  # Invalid: variable 2 has parent 2 (itself)
  parents_bad <- list(integer(0), c(2), c(1))

  expect_error(check_dag_order(parents_bad), "Invalid topological order")
})

test_that("check_dag_order rejects negative indices", {
  parents_bad <- list(integer(0), c(-1), c(1))

  expect_error(check_dag_order(parents_bad), "indices must be >= 1")
})

test_that("get_topological_order returns identity for valid order", {
  parents <- list(integer(0), c(1), c(1, 2))

  expect_equal(get_topological_order(parents), c(1, 2, 3))
})

test_that("print_dag runs without error", {
  parents <- make_longitudinal_dag(2, 2, "markov")

  # Capture output to verify it runs
  output <- capture.output(print_dag(parents))
  expect_true(length(output) > 0)
  expect_true(any(grepl("Z1_1", output)))
})
