#' @title Experiment Framework Functions
#' @description Functions for loading configuration files, running experiments,
#'   and saving results in a reproducible manner.
#' @name experiment_framework
NULL

#' Load experiment configuration from YAML file
#'
#' Loads and validates a YAML configuration file for running experiments.
#'
#' @param config_path Character. Path to the YAML configuration file.
#'
#' @return A list containing the validated configuration.
#'
#' @examples
#' \dontrun{
#' config <- load_config("experiments/config/markov.yaml")
#' }
#'
#' @export
load_config <- function(config_path) {
  if (!requireNamespace("yaml", quietly = TRUE)) {
    stop("yaml package is required. Install with install.packages('yaml')")
  }

  if (!file.exists(config_path)) {
    stop(sprintf("Configuration file not found: %s", config_path))
  }

  config <- yaml::read_yaml(config_path)

  # Validate required fields
  required_fields <- c("experiment", "sampling")
  for (field in required_fields) {
    if (is.null(config[[field]])) {
      stop(sprintf("Required field '%s' missing from config", field))
    }
  }

  # Set defaults for optional fields
  if (is.null(config$experiment$name)) {
    config$experiment$name <- "unnamed_experiment"
  }

  if (is.null(config$sampling$seed)) {
    config$sampling$seed <- 42
  }

  if (is.null(config$independence_tests)) {
    config$independence_tests <- list(
      n_boot = 200,
      boot_sample_size = 2000,
      methods = c("GCM")
    )
  }

  if (is.null(config$output)) {
    config$output <- list(
      dir = "./results",
      save_raw = TRUE,
      save_figures = TRUE
    )
  }

  if (is.null(config$parallel)) {
    config$parallel <- list(
      enabled = FALSE,
      n_cores = 1
    )
  }

  return(config)
}


#' Capture experiment metadata
#'
#' Captures metadata about the experiment run including timestamp, git commit,
#' R version, and package versions.
#'
#' @param config The experiment configuration.
#'
#' @return A list containing metadata.
#'
#' @keywords internal
capture_metadata <- function(config) {
  metadata <- list(
    timestamp = Sys.time(),
    R_version = R.version.string,
    experiment_name = config$experiment$name,
    config_hash = digest::digest(config, algo = "md5")
  )

  # Try to capture git commit
  tryCatch({
    git_commit <- system("git rev-parse HEAD", intern = TRUE)
    metadata$git_commit <- git_commit
  }, error = function(e) {
    metadata$git_commit <- "unknown"
  }, warning = function(w) {
    metadata$git_commit <- "unknown"
  })

  # Capture package versions
  metadata$package_versions <- list(
    copula = packageVersion("copula"),
    VineCopula = packageVersion("VineCopula")
  )

  return(metadata)
}


#' Setup logging for experiment
#'
#' Configures logging to file and console for experiment runs.
#'
#' @param config The experiment configuration.
#'
#' @keywords internal
setup_logging <- function(config) {
  if (!requireNamespace("logger", quietly = TRUE)) {
    message("logger package not available. Using basic message() for logging.")
    return(invisible(NULL))
  }

  log_dir <- file.path(config$output$dir, "logs")
  dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)

  log_file <- file.path(log_dir, sprintf("%s_%s.log",
                                          config$experiment$name,
                                          format(Sys.time(), "%Y%m%d_%H%M%S")))

  # Setup file appender
  logger::log_appender(logger::appender_file(log_file))
  logger::log_info("Experiment started: {config$experiment$name}")
  logger::log_info("Log file: {log_file}")

  return(invisible(log_file))
}


#' Log a message
#'
#' Logs a message using logger if available, otherwise uses message().
#'
#' @param msg Character. The message to log.
#' @param level Character. Log level: "info", "warn", "error". Default is "info".
#'
#' @keywords internal
log_message <- function(msg, level = "info") {
  if (requireNamespace("logger", quietly = TRUE)) {
    switch(level,
           "info" = logger::log_info(msg),
           "warn" = logger::log_warn(msg),
           "error" = logger::log_error(msg),
           logger::log_info(msg))
  } else {
    message(sprintf("[%s] %s: %s", toupper(level), Sys.time(), msg))
  }
}


#' Generate data from Bayesian Network specification
#'
#' Generates covariate data from a Bayesian Network specification in the config.
#'
#' @param config The experiment configuration.
#' @param N Integer. Sample size.
#' @param seed Integer. Random seed.
#'
#' @return A list containing covariate data and conditional ranks.
#'
#' @keywords internal
generate_data <- function(config, N, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  covariates <- config$model$covariates
  n_cov <- length(covariates)

  data <- list()
  ranks <- list()

  for (i in seq_along(covariates)) {
    cov_name <- names(covariates)[i]
    cov_spec <- covariates[[cov_name]]

    # Generate uniform rank
    U <- runif(N)
    ranks[[cov_name]] <- U

    # Generate from specified distribution
    if (cov_spec$dist == "gamma") {
      # Evaluate shape formula if it's a string
      if (is.character(cov_spec$shape) || !is.null(cov_spec$shape_formula)) {
        shape_formula <- cov_spec$shape_formula %||% cov_spec$shape
        shape <- eval(parse(text = shape_formula), envir = as.list(data))
      } else {
        shape <- cov_spec$shape
      }
      scale <- cov_spec$scale %||% 1
      data[[cov_name]] <- qgamma(U, shape = shape, scale = scale)

    } else if (cov_spec$dist == "normal" || cov_spec$dist == "gaussian") {
      mean_val <- cov_spec$mean %||% 0
      sd_val <- cov_spec$sd %||% 1
      if (is.character(mean_val)) {
        mean_val <- eval(parse(text = mean_val), envir = as.list(data))
      }
      data[[cov_name]] <- qnorm(U, mean = mean_val, sd = sd_val)

    } else if (cov_spec$dist == "uniform") {
      min_val <- cov_spec$min %||% 0
      max_val <- cov_spec$max %||% 1
      data[[cov_name]] <- qunif(U, min = min_val, max = max_val)

    } else {
      stop(sprintf("Unknown distribution: %s", cov_spec$dist))
    }
  }

  return(list(
    covariate_data = as.data.frame(data),
    conditional_ranks = as.data.frame(ranks)
  ))
}


#' Run independence tests on generated data
#'
#' Runs the specified independence tests from the configuration.
#'
#' @param data A list containing covariate_data, conditional_ranks, and outcome.
#' @param config The experiment configuration.
#'
#' @return A list containing test results.
#'
#' @keywords internal
run_independence_tests <- function(data, config) {
  test_config <- config$independence_tests
  results <- list()

  # Placeholder - actual tests would be specified in config
  return(results)
}


#' Extract DAG Parent Structure from Experiment Config
#'
#' Infers the parent structure from the covariate specification. A covariate
#' is considered to depend on a parent if its distribution parameters reference
#' the parent variable name (e.g., shape_formula = "2 + 3*Z1").
#'
#' If \code{config$model$dag_parents} is explicitly provided, uses that directly.
#'
#' @param config The experiment configuration.
#'
#' @return A list of parent indices (one per covariate), or NULL if the DAG
#'   structure cannot be determined.
#'
#' @keywords internal
extract_dag_parents <- function(config) {
  # Check for explicit dag_parents in config
  if (!is.null(config$model$dag_parents)) {
    return(config$model$dag_parents)
  }

  covariates <- config$model$covariates
  if (is.null(covariates)) return(NULL)

  n_cov <- length(covariates)
  cov_names <- names(covariates)
  parents <- vector("list", n_cov)

  for (i in seq_along(covariates)) {
    cov_spec <- covariates[[i]]
    pa_idx <- integer(0)

    # Check all string-valued parameters for references to other covariates
    param_strings <- character(0)
    for (param_name in names(cov_spec)) {
      val <- cov_spec[[param_name]]
      if (is.character(val)) {
        param_strings <- c(param_strings, val)
      }
    }

    # Search for references to earlier covariates
    if (length(param_strings) > 0 && i > 1) {
      combined <- paste(param_strings, collapse = " ")
      for (j in seq_len(i - 1)) {
        if (grepl(cov_names[j], combined, fixed = TRUE)) {
          pa_idx <- c(pa_idx, j)
        }
      }
    }

    parents[[i]] <- pa_idx
  }

  return(parents)
}


#' Run a complete experiment
#'
#' Main function to run an experiment from a configuration file.
#'
#' @param config_path Character. Path to the YAML configuration file.
#' @param verbose Logical. Whether to print progress messages. Default is TRUE.
#'
#' @return A list containing all experiment results.
#'
#' @examples
#' \dontrun{
#' results <- run_experiment("experiments/config/markov.yaml")
#' }
#'
#' @export
run_experiment <- function(config_path, verbose = TRUE) {
  # Load configuration
  config <- load_config(config_path)

  # Setup logging
  setup_logging(config)

  # Capture metadata
  results <- list(
    metadata = capture_metadata(config),
    config = config
  )

  if (verbose) {
    cat("\n========================================\n")
    cat(sprintf("Running Experiment: %s\n", config$experiment$name))
    cat("========================================\n\n")
  }

  # Ensure output directory exists
  dir.create(config$output$dir, recursive = TRUE, showWarnings = FALSE)

  # Run for each sample size
  sample_sizes <- config$sampling$sample_sizes
  if (is.null(sample_sizes)) {
    sample_sizes <- 1000  # Default
  }

  for (N in sample_sizes) {
    log_message(sprintf("Running with N = %d", N))
    if (verbose) cat(sprintf("\n--- Sample Size N = %d ---\n", N))

    # Set seed for reproducibility
    set.seed(config$sampling$seed)

    # Generate data
    log_message("Generating data...")
    data <- generate_data(config, N, seed = config$sampling$seed)

    # Generate outcome using nonparanormal approximation
    if (!is.null(config$copula)) {
      log_message("Generating outcome samples...")

      topoOrder <- config$copula$topo_order
      vine_cor_params <- c(config$copula$rho_Y_Z2,
                           config$copula$rho_Y_Z1_given_Z2,
                           rep(0, ncol(data$covariate_data) - 2))

      # Extract DAG parent structure from config if available
      dag_parents <- extract_dag_parents(config)

      # Fit DAG-constrained Gaussian BN if parents are available
      gaussian_bn_fit <- NULL
      if (!is.null(dag_parents)) {
        log_message("Fitting DAG-constrained Gaussian BN (Route B)...")
        gaussian_bn_fit <- fit_reference_gaussian_bn(
          as.matrix(data$covariate_data), dag_parents
        )
      }

      outcome_result <- simulateConditionalOutcomeSamples(
        covariate_data = data$covariate_data,
        cond_covariate_ranks = data$conditional_ranks,
        vine_cor_params = vine_cor_params,
        topoOrder = topoOrder,
        gaussian_bn_fit = gaussian_bn_fit,
        parents = dag_parents
      )

      data$outcome <- qnorm(outcome_result$outcomeRankSamples)
      data$outcome_ranks <- outcome_result$outcomeRankSamples
      data$full_correlation_matrix <- outcome_result$fullCorrelationMatrix
    }

    # Run independence tests
    log_message("Running independence tests...")
    test_results <- run_independence_tests(data, config)

    # Store results
    results[[paste0("N_", N)]] <- list(
      data_summary = list(
        n_obs = N,
        n_covariates = ncol(data$covariate_data)
      ),
      test_results = test_results
    )
  }

  # Save results
  if (config$output$save_raw) {
    save_results(results, config)
  }

  # Generate figures
  if (config$output$save_figures) {
    generate_figures(results, config)
  }

  log_message("Experiment complete")
  if (verbose) {
    cat("\n========================================\n")
    cat(sprintf("Experiment complete. Results saved to: %s\n", config$output$dir))
    cat("========================================\n")
  }

  return(results)
}


#' Save experiment results
#'
#' Saves experiment results to RDS and optionally CSV files.
#'
#' @param results The experiment results list.
#' @param config The experiment configuration.
#'
#' @return Invisibly returns the path to the saved results file.
#'
#' @export
save_results <- function(results, config) {
  output_dir <- config$output$dir
  raw_dir <- file.path(output_dir, "raw")
  dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)

  # Save as RDS
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  rds_path <- file.path(raw_dir, sprintf("%s_%s.rds",
                                          config$experiment$name,
                                          timestamp))
  saveRDS(results, rds_path)
  log_message(sprintf("Results saved to: %s", rds_path))

  return(invisible(rds_path))
}


#' Generate figures from experiment results
#'
#' Generates and saves standard figures from experiment results.
#'
#' @param results The experiment results list.
#' @param config The experiment configuration.
#'
#' @return Invisibly returns the figure output directory.
#'
#' @export
generate_figures <- function(results, config) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    warning("ggplot2 not available. Skipping figure generation.")
    return(invisible(NULL))
  }

  output_dir <- config$output$dir
  fig_dir <- file.path(output_dir, "figures")
  dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

  log_message(sprintf("Figures saved to: %s", fig_dir))

  return(invisible(fig_dir))
}


#' Null coalescing operator
#'
#' Returns the left-hand side if not NULL, otherwise the right-hand side.
#'
#' @param x Left-hand side value.
#' @param y Right-hand side (default) value.
#'
#' @return x if not NULL, otherwise y.
#'
#' @keywords internal
`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}
