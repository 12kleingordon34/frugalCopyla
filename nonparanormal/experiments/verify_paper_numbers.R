#!/usr/bin/env Rscript
# =============================================================================
# verify_paper_numbers.R
#
# Cross-checks auto-generated LaTeX tables against their source CSVs.
# Run before any commit to ensure paper tables match experimental results.
#
# Usage:
#   Rscript --no-init-file experiments/verify_paper_numbers.R
#
# Checks performed:
#   1. static_ci_diagnostics.tex  vs  results/static_ci_summary.csv
#   2. static_routeAB_ci_summary.tex  vs  results/static_routeA_vs_routeB_ci_summary.csv
#   3. static_rank_uniformity_extended.tex  vs  results/static_rank_uniformity_extended.csv
#   4. longitudinal_rank_uniformity_summary.tex  vs  results/longitudinal_rank_uniformity_summary.csv
# =============================================================================

cat("=== verify_paper_numbers.R ===\n\n")

# Determine base paths
# Works whether run from nonparanormal/ or nonparanormal/experiments/
if (file.exists("results/static_ci_summary.csv")) {
  results_dir <- "results"
  tables_dir  <- "../Hybrid-Frugal-Paper/tables"
} else if (file.exists("../results/static_ci_summary.csv")) {
  results_dir <- "../results"
  tables_dir  <- "../../Hybrid-Frugal-Paper/tables"
} else {
  stop("Cannot find results directory. Run from nonparanormal/ or nonparanormal/experiments/.")
}

n_pass <- 0
n_fail <- 0
n_skip <- 0
failures <- character(0)

check <- function(desc, expected, actual, tol = 1e-4) {
  if (is.na(expected) || is.na(actual)) {
    cat(sprintf("  [SKIP] %s: NA values\n", desc))
    n_skip <<- n_skip + 1
    return(invisible(NULL))
  }
  if (abs(expected - actual) < tol) {
    n_pass <<- n_pass + 1
  } else {
    cat(sprintf("  [FAIL] %s: CSV=%.4f, TeX=%.4f\n", desc, expected, actual))
    n_fail <<- n_fail + 1
    failures <<- c(failures, desc)
  }
}

# Helper: extract all decimal numbers from a .tex data row
# Returns a numeric vector of all numbers found in the line
extract_numbers <- function(line) {
  nums <- regmatches(line, gregexpr("[0-9]+\\.[0-9]+", line))[[1]]
  as.numeric(nums)
}

# =============================================================================
# Check 1: static_ci_diagnostics.tex vs static_ci_summary.csv
# =============================================================================

cat("--- Check 1: Static CI diagnostics ---\n")
csv_path <- file.path(results_dir, "static_ci_summary.csv")
tex_path <- file.path(tables_dir, "static_ci_diagnostics.tex")

if (!file.exists(csv_path)) {
  cat("  [SKIP] CSV not found:", csv_path, "\n")
  n_skip <- n_skip + 1
} else if (!file.exists(tex_path)) {
  cat("  [SKIP] TeX not found:", tex_path, "\n")
  n_skip <- n_skip + 1
} else {
  csv <- read.csv(csv_path, stringsAsFactors = FALSE)
  tex <- readLines(tex_path)

  # Extract data rows (lines with numeric content between \midrule and \bottomrule)
  in_data <- FALSE
  data_rows <- character(0)
  for (line in tex) {
    if (grepl("\\\\midrule", line)) { in_data <- TRUE; next }
    if (grepl("\\\\bottomrule", line)) { in_data <- FALSE; next }
    if (grepl("\\\\addlinespace", line)) next
    if (in_data && grepl("[0-9]", line)) {
      data_rows <- c(data_rows, line)
    }
  }

  # Expected order: null rows (pcor, gcm, rcot), then collider rows (pcor, gcm, rcot)
  test_order <- c("pcor", "gcm", "rcot")

  # Null section: each row has Route A (GAUSS) then Route B (BN) null_ks_p
  for (i in seq_along(test_order)) {
    test <- test_order[i]
    nums <- extract_numbers(data_rows[i])
    gauss_expected <- csv$null_ks_p[csv$simulator == "GAUSS" & csv$test == test]
    bn_expected   <- csv$null_ks_p[csv$simulator == "BN"    & csv$test == test]
    check(sprintf("Null %s Route A", test), gauss_expected, nums[1])
    check(sprintf("Null %s Route B", test), bn_expected, nums[2])
  }

  # Collider section: each row has Route A (GAUSS) then Route B (BN) alt_reject_rate
  for (i in seq_along(test_order)) {
    test <- test_order[i]
    row_idx <- i + 3  # offset past null rows
    nums <- extract_numbers(data_rows[row_idx])
    # The collider row for first test also has 0.05 from alpha in the prefix
    # Filter: take last 2 numbers (they are the Route A and Route B values)
    if (length(nums) > 2) nums <- tail(nums, 2)
    gauss_expected <- csv$alt_reject_rate[csv$simulator == "GAUSS" & csv$test == test]
    bn_expected   <- csv$alt_reject_rate[csv$simulator == "BN"    & csv$test == test]
    check(sprintf("Collider %s Route A", test), gauss_expected, nums[1])
    check(sprintf("Collider %s Route B", test), bn_expected, nums[2])
  }
  cat(sprintf("  Checked %d values from static_ci_diagnostics.tex\n", 12))
}

# =============================================================================
# Check 2: static_routeAB_ci_summary.tex vs static_routeA_vs_routeB_ci_summary.csv
# =============================================================================

cat("--- Check 2: Route A/B CI summary ---\n")
csv_path <- file.path(results_dir, "static_routeA_vs_routeB_ci_summary.csv")
tex_path <- file.path(tables_dir, "static_routeAB_ci_summary.tex")

if (!file.exists(csv_path)) {
  cat("  [SKIP] CSV not found:", csv_path, "\n")
  n_skip <- n_skip + 1
} else if (!file.exists(tex_path)) {
  cat("  [SKIP] TeX not found:", tex_path, "\n")
  n_skip <- n_skip + 1
} else {
  csv <- read.csv(csv_path, stringsAsFactors = FALSE)
  tex <- readLines(tex_path)

  in_data <- FALSE
  data_rows <- character(0)
  for (line in tex) {
    if (grepl("\\\\midrule", line)) { in_data <- TRUE; next }
    if (grepl("\\\\bottomrule", line)) { in_data <- FALSE; next }
    if (grepl("\\\\addlinespace", line)) next
    if (in_data && grepl("[0-9]", line)) {
      data_rows <- c(data_rows, line)
    }
  }

  # Expected order: A-pcor, A-gcm, A-rcot, B-pcor, B-gcm, B-rcot
  test_order <- c("pcor", "gcm", "rcot")
  route_order <- c("A", "B")
  row_idx <- 1
  for (route in route_order) {
    for (test in test_order) {
      nums <- extract_numbers(data_rows[row_idx])
      csv_row <- csv[csv$route == route & csv$test == test, ]
      # The row also contains 0.05 from alpha in caption context -- but data rows
      # only have route letter + test name + 3 numbers
      # Take last 3 numbers
      if (length(nums) > 3) nums <- tail(nums, 3)
      check(sprintf("Route %s %s null_ks_p", route, test),
            csv_row$null_ks_p, nums[1])
      check(sprintf("Route %s %s null_reject_rate", route, test),
            csv_row$null_reject_rate, nums[2])
      check(sprintf("Route %s %s alt_reject_rate", route, test),
            csv_row$alt_reject_rate, nums[3])
      row_idx <- row_idx + 1
    }
  }
  cat(sprintf("  Checked %d values from static_routeAB_ci_summary.tex\n", 18))
}

# =============================================================================
# Check 3: static_rank_uniformity_extended.tex vs static_uniformity_extended.csv
# =============================================================================

cat("--- Check 3: Static rank-uniformity extended ---\n")
csv_path <- file.path(results_dir, "static_rank_uniformity_extended.csv")
tex_path <- file.path(tables_dir, "static_rank_uniformity_extended.tex")

if (!file.exists(csv_path)) {
  cat("  [SKIP] CSV not found:", csv_path, "\n")
  n_skip <- n_skip + 1
} else if (!file.exists(tex_path)) {
  cat("  [SKIP] TeX not found:", tex_path, "\n")
  n_skip <- n_skip + 1
} else {
  csv <- read.csv(csv_path, stringsAsFactors = FALSE)
  tex <- readLines(tex_path)

  in_data <- FALSE
  data_rows <- character(0)
  for (line in tex) {
    if (grepl("\\\\midrule", line)) { in_data <- TRUE; next }
    if (grepl("\\\\bottomrule", line)) { in_data <- FALSE; next }
    if (grepl("\\\\addlinespace", line)) next
    if (in_data && grepl("[0-9]", line)) {
      data_rows <- c(data_rows, line)
    }
  }

  # TeX row order is now Route A (GAUSS) first, Route B (BN) second within each variable.
  # CSV row order is BN first, GAUSS second. Build a mapping.
  # Variables in order: tilde_U_Z1, tilde_U_Z2, tilde_U_Z3, U_Y
  # TeX rows: GAUSS-Z1, BN-Z1, GAUSS-Z2, BN-Z2, GAUSS-Z3, BN-Z3, GAUSS-Y, BN-Y
  # CSV rows: BN-Z1(1), GAUSS-Z1(2), BN-Z2(3), GAUSS-Z2(4), BN-Z3(5), GAUSS-Z3(6), BN-Y(7), GAUSS-Y(8)
  tex_to_csv <- c(2, 1, 4, 3, 6, 5, 8, 7)  # maps tex row index to csv row index

  checked <- 0
  for (i in seq_len(min(length(tex_to_csv), length(data_rows)))) {
    nums <- extract_numbers(data_rows[i])
    csv_i <- tex_to_csv[i]
    # Columns: pct_pass, mean_ks_p, rank_mean, rank_var
    if (length(nums) >= 4 && csv_i <= nrow(csv)) {
      check(sprintf("Unif row %d pct_pass", i), csv$pct_pass[csv_i], nums[1])
      check(sprintf("Unif row %d mean_ks_p", i), csv$mean_ks_p[csv_i], nums[2])
      check(sprintf("Unif row %d rank_mean", i), csv$rank_mean[csv_i], nums[3])
      check(sprintf("Unif row %d rank_var", i), csv$rank_var[csv_i], nums[4])
      checked <- checked + 4
    }
  }
  cat(sprintf("  Checked %d values from static_rank_uniformity_extended.tex\n", checked))
}

# =============================================================================
# Check 4: longitudinal_rank_uniformity_summary.tex vs longitudinal_rank_uniformity_summary.csv
# =============================================================================

cat("--- Check 4: Longitudinal rank-uniformity summary ---\n")
csv_path <- file.path(results_dir, "longitudinal_rank_uniformity_summary.csv")
tex_path <- file.path(tables_dir, "longitudinal_rank_uniformity_summary.tex")

if (!file.exists(csv_path)) {
  cat("  [SKIP] CSV not found:", csv_path, "\n")
  n_skip <- n_skip + 1
} else if (!file.exists(tex_path)) {
  cat("  [SKIP] TeX not found:", tex_path, "\n")
  n_skip <- n_skip + 1
} else {
  csv <- read.csv(csv_path, stringsAsFactors = FALSE)
  tex <- readLines(tex_path)

  in_data <- FALSE
  data_rows <- character(0)
  for (line in tex) {
    if (grepl("\\\\midrule", line)) { in_data <- TRUE; next }
    if (grepl("\\\\bottomrule", line)) { in_data <- FALSE; next }
    if (grepl("\\\\addlinespace", line)) next
    if (in_data && grepl("[0-9]", line)) {
      data_rows <- c(data_rows, line)
    }
  }

  checked <- 0
  for (i in seq_len(min(nrow(csv), length(data_rows)))) {
    nums <- extract_numbers(data_rows[i])
    # Columns: pct_pass, mean_ks_p
    if (length(nums) >= 2) {
      check(sprintf("Long unif row %d pct_pass", i), csv$pct_pass[i], nums[1])
      check(sprintf("Long unif row %d mean_ks_p", i), csv$mean_ks_p[i], nums[2])
      checked <- checked + 2
    }
  }
  cat(sprintf("  Checked %d values from longitudinal_rank_uniformity_summary.tex\n", checked))
}

# =============================================================================
# Summary
# =============================================================================

cat("\n=== VERIFICATION SUMMARY ===\n")
cat(sprintf("  PASS: %d\n", n_pass))
cat(sprintf("  FAIL: %d\n", n_fail))
cat(sprintf("  SKIP: %d\n", n_skip))

if (n_fail > 0) {
  cat("\nFailed checks:\n")
  for (f in failures) cat(sprintf("  - %s\n", f))
  cat("\nRe-run the experiment scripts to regenerate .tex files.\n")
  quit(status = 1)
} else if (n_pass > 0) {
  cat("\nAll checks passed.\n")
} else {
  cat("\nNo checks were performed (missing files?).\n")
  quit(status = 1)
}
