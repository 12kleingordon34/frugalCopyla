#!/usr/bin/env Rscript
# =============================================================================
# Setup renv for nonparanormal package
# =============================================================================
#
# Run this script to initialize renv and install all required dependencies.
#
# Usage:
#   Rscript setup_renv.R
#
# =============================================================================

cat("Setting up renv for nonparanormal package...\n\n")

# Install renv if not available
if (!requireNamespace("renv", quietly = TRUE)) {
  cat("Installing renv...\n")
  install.packages("renv")
}

# Initialize renv
cat("Initializing renv...\n")
renv::init(bare = TRUE)

# Define required packages
packages <- c(
  # Core copula packages
  "copula",
  "VineCopula",

  # Independence testing
  "GeneralisedCovarianceMeasure",
  "CondIndTests",
  "bnlearn",

  # Statistical utilities
  "ppcor",
  "MASS",

  # Visualization
  "ggplot2",
  "gridExtra",
  "tidyverse",

  # Testing
  "testthat",

  # Experiment framework
  "yaml",
  "logger",
  "digest",

  # Parallelization
  "future",
  "future.apply",

  # Development utilities
  "devtools",
  "roxygen2",
  "usethis"
)

# Install packages
cat("\nInstalling packages...\n")
for (pkg in packages) {
  cat(sprintf("  Installing %s...\n", pkg))
  tryCatch({
    renv::install(pkg)
  }, error = function(e) {
    cat(sprintf("    Warning: Could not install %s: %s\n", pkg, e$message))
  })
}

# Create snapshot
cat("\nCreating renv snapshot...\n")
renv::snapshot()

cat("\n=== Setup Complete ===\n")
cat("Run 'renv::restore()' to restore the environment in the future.\n")
cat("Run 'devtools::test()' to run the test suite.\n")
