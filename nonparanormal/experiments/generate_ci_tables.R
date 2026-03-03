#!/usr/bin/env Rscript
# =============================================================================
# generate_ci_tables.R
#
# Generates LaTeX CI diagnostics tables from existing summary CSVs.
# This is the same code embedded at the end of static_clayton_vine.R and
# static_routeA_vs_routeB.R, extracted here so tables can be regenerated
# without re-running the full experiments.
#
# Usage:
#   Rscript --no-init-file experiments/generate_ci_tables.R
# =============================================================================

cat("=== generate_ci_tables.R ===\n\n")

# Determine base paths
if (file.exists("results/static_ci_summary.csv")) {
  results_dir <- "results"
  tables_dir  <- "../Hybrid-Frugal-Paper/tables"
} else if (file.exists("../results/static_ci_summary.csv")) {
  results_dir <- "../results"
  tables_dir  <- "../../Hybrid-Frugal-Paper/tables"
} else {
  stop("Cannot find results directory. Run from nonparanormal/ or nonparanormal/experiments/.")
}

if (!dir.exists(tables_dir)) dir.create(tables_dir, recursive = TRUE)

# =============================================================================
# Table 1: static_ci_diagnostics.tex (main text)
# =============================================================================

csv_path <- file.path(results_dir, "static_ci_summary.csv")
if (!file.exists(csv_path)) {
  cat("SKIP: ", csv_path, " not found\n")
} else {
  ci_summary <- read.csv(csv_path, stringsAsFactors = FALSE)
  test_labels <- c(pcor = "pcor", gcm = "GCM", rcot = "RCoT")

  tex_lines <- c(
    "\\begin{table}[htbp]",
    "\\centering",
    paste0(
      "\\caption{CI diagnostics for model $\\mathcal{M}_A$ (\\Cref{fig:MA-dag}), ",
      "based on 200 Monte Carlo replications with $N=1000$ each. ",
      "For the null relation $Z_2 \\indep Y \\mid (Z_1,Z_3)$ we report the KS $p$-value ",
      "for uniformity of test $p$-values. For the collider diagnostic ",
      "$Z_1 \\nindep Z_3 \\mid (Y,Z_2)$ we report the empirical rejection rate ",
      "at level $\\alpha=0.05$.}"
    ),
    "\\label{tab:ci-tests-MA}",
    "\\begin{tabularx}{\\linewidth}{@{}p{0.44\\linewidth}p{0.24\\linewidth}cc@{}}",
    "\\toprule",
    "\\textbf{Case / Metric} & \\textbf{CI diagnostic} & \\textbf{BN (B)} & \\textbf{GAUSS (A)} \\\\",
    "\\midrule"
  )

  # Null rows
  for (i in seq_along(test_labels)) {
    test_name <- names(test_labels)[i]
    label <- test_labels[i]
    bn_val   <- ci_summary$null_ks_p[ci_summary$simulator == "BN"   & ci_summary$test == test_name]
    gauss_val <- ci_summary$null_ks_p[ci_summary$simulator == "GAUSS" & ci_summary$test == test_name]
    prefix <- if (i == 1) "\\textit{Null / KS $p$-value}" else ""
    tex_lines <- c(tex_lines, sprintf(
      "%s & %s & %.3f & %.3f \\\\", prefix, label, bn_val, gauss_val
    ))
  }

  tex_lines <- c(tex_lines, "\\addlinespace")

  # Collider rows
  for (i in seq_along(test_labels)) {
    test_name <- names(test_labels)[i]
    label <- test_labels[i]
    bn_val   <- ci_summary$alt_reject_rate[ci_summary$simulator == "BN"   & ci_summary$test == test_name]
    gauss_val <- ci_summary$alt_reject_rate[ci_summary$simulator == "GAUSS" & ci_summary$test == test_name]
    prefix <- if (i == 1) "\\textit{Collider / Rejection rate at $\\alpha=0.05$}" else ""
    tex_lines <- c(tex_lines, sprintf(
      "%s & %s & %.3f & %.3f \\\\", prefix, label, bn_val, gauss_val
    ))
  }

  tex_lines <- c(tex_lines,
    "\\bottomrule",
    "\\end{tabularx}",
    "\\end{table}"
  )

  tex_path <- file.path(tables_dir, "static_ci_diagnostics.tex")
  writeLines(tex_lines, tex_path)
  cat(sprintf("Saved %s\n", tex_path))
}

# =============================================================================
# Table 2: static_routeAB_ci_summary.tex (appendix)
# =============================================================================

csv_path <- file.path(results_dir, "static_routeA_vs_routeB_ci_summary.csv")
if (!file.exists(csv_path)) {
  cat("SKIP: ", csv_path, " not found\n")
} else {
  ci_summary <- read.csv(csv_path, stringsAsFactors = FALSE)
  test_labels <- c(pcor = "Partial cor.\\ ", gcm = "GCM", rcot = "RCoT")

  tex_lines <- c(
    "\\begin{table}[htbp]",
    "\\centering",
    paste0(
      "\\caption{Route A vs Route B: CI test summary for the static $\\mathcal{M}_A$ model. ",
      "Null KS $p$: Kolmogorov--Smirnov test for uniformity of null p-values (larger is better). ",
      "The null rejection rate represents fraction of replications rejecting ",
      "$Z_2 \\perp\\!\\!\\!\\perp Y \\mid Z_1, Z_3$ at $\\alpha = 0.05$ (target $\\approx 0.05$). ",
      "The collider rejection rate represents fraction rejecting ",
      "$Z_1 \\perp\\!\\!\\!\\perp Z_3 \\mid Y, Z_2$, with larger values indicating greater power. }"
    ),
    "\\label{tab:routeAB-ci-summary}",
    "\\begin{tabular}{llccc}",
    "\\toprule",
    "\\textbf{Route} & \\textbf{Test} & \\textbf{Null KS $p$} & \\textbf{Null rej.\\ rate} & \\textbf{Collider rej.\\ rate} \\\\",
    "\\midrule"
  )

  for (route in c("A", "B")) {
    route_data <- ci_summary[ci_summary$route == route, ]
    for (i in seq_along(test_labels)) {
      test_name <- names(test_labels)[i]
      label <- test_labels[i]
      row <- route_data[route_data$test == test_name, ]
      tex_lines <- c(tex_lines, sprintf(
        "%s & %s & %.3f & %.3f & %.3f \\\\",
        route, label, row$null_ks_p, row$null_reject_rate, row$alt_reject_rate
      ))
    }
    if (route == "A") {
      tex_lines <- c(tex_lines, "\\addlinespace")
    }
  }

  tex_lines <- c(tex_lines,
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table}"
  )

  tex_path <- file.path(tables_dir, "static_routeAB_ci_summary.tex")
  writeLines(tex_lines, tex_path)
  cat(sprintf("Saved %s\n", tex_path))
}

cat("\nDone.\n")
