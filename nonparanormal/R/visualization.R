#' @title Visualization Functions
#' @description Functions for creating standardized plots for nonparanormal
#'   experiment results.
#' @name visualization
NULL

#' Standard theme for nonparanormal plots
#'
#' Returns a ggplot2 theme with consistent styling for all package plots.
#'
#' @return A ggplot2 theme object.
#'
#' @examples
#' \dontrun{
#' library(ggplot2)
#' ggplot(data.frame(x = rnorm(100)), aes(x = x)) +
#'   geom_histogram() +
#'   theme_nonparanormal()
#' }
#'
#' @export
theme_nonparanormal <- function() {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("ggplot2 package is required for visualization functions.")
  }

  ggplot2::theme_minimal() +
    ggplot2::theme(
      text = ggplot2::element_text(family = "Helvetica", size = 14),
      axis.title = ggplot2::element_text(face = "bold"),
      panel.grid.major = ggplot2::element_line(color = "grey90"),
      panel.grid.minor = ggplot2::element_line(color = "grey95"),
      panel.background = ggplot2::element_rect(fill = "white"),
      plot.background = ggplot2::element_rect(fill = "white"),
      panel.border = ggplot2::element_blank(),
      plot.title = ggplot2::element_text(hjust = 0.5, face = "bold")
    )
}


#' P-value histogram with uniform reference line
#'
#' Creates a histogram of p-values with a reference line at the expected
#' count under the uniform null hypothesis.
#'
#' @param pvalues A numeric vector of p-values.
#' @param title Character. Plot title. Default is NULL (no title).
#' @param fill_color Character. Fill color for histogram bars. Default is "steelblue".
#' @param n_bins Integer. Number of histogram bins. Default is 10.
#' @param add_ks_test Logical. Whether to add KS test result to subtitle. Default is TRUE.
#'
#' @return A ggplot2 object.
#'
#' @examples
#' \dontrun{
#' pvals <- runif(200)  # Under null, should be uniform
#' plot_pvalue_histogram(pvals, title = "Test p-values")
#' }
#'
#' @export
plot_pvalue_histogram <- function(pvalues, title = NULL, fill_color = "steelblue",
                                   n_bins = 10, add_ks_test = TRUE) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("ggplot2 package is required for visualization functions.")
  }

  # Remove NAs
  pvalues <- pvalues[!is.na(pvalues)]

  # Compute KS test for uniformity
  ks_result <- stats::ks.test(pvalues, "punif")

  # Create subtitle with KS test result
  subtitle <- NULL
  if (add_ks_test) {
    subtitle <- sprintf("KS test for uniformity: p = %.4f", ks_result$p.value)
  }

  # Expected count per bin under uniform null
  expected_count <- length(pvalues) / n_bins

  # Create the plot
  p <- ggplot2::ggplot(data.frame(p_value = pvalues), ggplot2::aes(x = p_value)) +
    ggplot2::geom_histogram(bins = n_bins, fill = fill_color, colour = "black",
                            alpha = 0.7) +
    ggplot2::geom_hline(yintercept = expected_count, linetype = "dashed",
                        color = "red", linewidth = 0.8) +
    ggplot2::labs(title = title, subtitle = subtitle,
                  x = "p-value", y = "Frequency") +
    ggplot2::scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
    theme_nonparanormal()

  return(p)
}


#' Causal margin verification plot
#'
#' Creates a plot to verify that the causal margin is preserved by comparing
#' the empirical distribution of residuals to the theoretical N(0, 1).
#'
#' @param residuals A numeric vector of residuals (Y - expected).
#' @param title Character. Plot title.
#' @param expected_mean Numeric. The expected mean. Default is 0.
#' @param expected_sd Numeric. The expected standard deviation. Default is 1.
#'
#' @return A ggplot2 object with a histogram and normal density overlay.
#'
#' @examples
#' \dontrun{
#' residuals <- rnorm(1000)
#' plot_causal_margin_check(residuals, title = "Residual Distribution")
#' }
#'
#' @export
plot_causal_margin_check <- function(residuals, title = "Causal Margin Check",
                                      expected_mean = 0, expected_sd = 1) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("ggplot2 package is required for visualization functions.")
  }

  # Remove NAs
  residuals <- residuals[!is.na(residuals)]

  # Compute empirical statistics
  emp_mean <- mean(residuals)
  emp_sd <- sd(residuals)

  # Compute KS test against theoretical distribution
  standardized <- (residuals - expected_mean) / expected_sd
  ks_result <- stats::ks.test(standardized, "pnorm")

  subtitle <- sprintf(
    "Empirical: mean=%.3f, sd=%.3f | Expected: mean=%.1f, sd=%.1f | KS p=%.4f",
    emp_mean, emp_sd, expected_mean, expected_sd, ks_result$p.value
  )

  # Create the plot
  p <- ggplot2::ggplot(data.frame(residuals = residuals), ggplot2::aes(x = residuals)) +
    ggplot2::geom_histogram(ggplot2::aes(y = ggplot2::after_stat(density)),
                            bins = 30, fill = "lightblue", colour = "black",
                            alpha = 0.7) +
    ggplot2::stat_function(fun = stats::dnorm, args = list(mean = expected_mean,
                                                           sd = expected_sd),
                           color = "red", linewidth = 1) +
    ggplot2::labs(title = title, subtitle = subtitle,
                  x = "Residuals", y = "Density") +
    theme_nonparanormal()

  return(p)
}


#' Generate experiment panel of plots
#'
#' Creates a multi-panel figure combining conditional independence tests,
#' marginal dependence tests, and causal margin verification.
#'
#' @param results A list containing experiment results with p-values and test statistics.
#' @param output_path Character. Path to save the combined figure.
#' @param width Numeric. Figure width in inches. Default is 12.
#' @param height Numeric. Figure height in inches. Default is 8.
#' @param dpi Integer. Resolution for saved figure. Default is 300.
#'
#' @return Invisibly returns the combined plot object.
#'
#' @examples
#' \dontrun{
#' results <- list(
#'   cond_ind_pvals = list(test1 = runif(100), test2 = runif(100)),
#'   marg_dep_pvals = list(test1 = runif(100), test2 = runif(100))
#' )
#' generate_experiment_panel(results, output_path = "results/figure.png")
#' }
#'
#' @export
generate_experiment_panel <- function(results, output_path = NULL,
                                       width = 12, height = 8, dpi = 300) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("ggplot2 package is required for visualization functions.")
  }
  if (!requireNamespace("gridExtra", quietly = TRUE)) {
    stop("gridExtra package is required for multi-panel plots.")
  }

  plot_list <- list()

  # Generate conditional independence plots
  if (!is.null(results$cond_ind_pvals)) {
    for (name in names(results$cond_ind_pvals)) {
      p <- plot_pvalue_histogram(
        results$cond_ind_pvals[[name]],
        title = paste("CI:", name),
        fill_color = "steelblue"
      )
      plot_list <- c(plot_list, list(p))
    }
  }

  # Generate marginal dependence plots
  if (!is.null(results$marg_dep_pvals)) {
    for (name in names(results$marg_dep_pvals)) {
      p <- plot_pvalue_histogram(
        results$marg_dep_pvals[[name]],
        title = paste("Marg:", name),
        fill_color = "coral"
      )
      plot_list <- c(plot_list, list(p))
    }
  }

  # Combine plots
  if (length(plot_list) > 0) {
    combined <- gridExtra::arrangeGrob(grobs = plot_list,
                                        ncol = min(2, length(plot_list)))

    # Save if output path provided
    if (!is.null(output_path)) {
      dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
      ggplot2::ggsave(output_path, plot = combined,
                      width = width, height = height, dpi = dpi)
    }

    return(invisible(combined))
  }

  return(invisible(NULL))
}


#' Create summary table of test results
#'
#' Creates a formatted summary table of KS test p-values for conditional
#' independence and marginal dependence tests.
#'
#' @param results A list containing experiment results.
#' @param format Character. Output format: "data.frame", "markdown", or "latex".
#'   Default is "data.frame".
#'
#' @return A data frame or character string depending on format.
#'
#' @examples
#' \dontrun{
#' results <- list(
#'   sample_sizes = c(1000, 5000),
#'   N_1000 = list(
#'     cond_ind_results = list(
#'       t2 = list(ks_Y_Z1prev = list(p.value = 0.15))
#'     )
#'   )
#' )
#' create_summary_table(results, format = "markdown")
#' }
#'
#' @export
create_summary_table <- function(results, format = "data.frame") {
  # Extract sample sizes
  sample_sizes <- results$sample_sizes
  if (is.null(sample_sizes)) {
    sample_sizes <- grep("^N_", names(results), value = TRUE)
    sample_sizes <- as.numeric(gsub("N_", "", sample_sizes))
  }

  # Build summary data frame
  summary_data <- data.frame()

  for (N in sample_sizes) {
    N_results <- results[[paste0("N_", N)]]
    if (is.null(N_results)) next

    # Extract conditional independence KS p-values
    cond_ind <- N_results$cond_ind_results
    if (!is.null(cond_ind)) {
      for (test_name in names(cond_ind)) {
        test_result <- cond_ind[[test_name]]
        for (metric in names(test_result)) {
          if (grepl("ks_", metric)) {
            p_val <- test_result[[metric]]$p.value
            if (!is.null(p_val)) {
              summary_data <- rbind(summary_data, data.frame(
                N = N,
                Test_Type = "Conditional Independence",
                Test = paste(test_name, metric, sep = "_"),
                KS_pvalue = p_val,
                stringsAsFactors = FALSE
              ))
            }
          }
        }
      }
    }
  }

  if (format == "markdown") {
    if (nrow(summary_data) == 0) return("")
    lines <- c(
      "| N | Test Type | Test | KS p-value |",
      "|---|-----------|------|------------|"
    )
    for (i in 1:nrow(summary_data)) {
      lines <- c(lines, sprintf("| %d | %s | %s | %.4f |",
                                summary_data$N[i],
                                summary_data$Test_Type[i],
                                summary_data$Test[i],
                                summary_data$KS_pvalue[i]))
    }
    return(paste(lines, collapse = "\n"))
  } else if (format == "latex") {
    if (nrow(summary_data) == 0) return("")
    lines <- c(
      "\\begin{tabular}{|c|c|c|c|}",
      "\\hline",
      "N & Test Type & Test & KS p-value \\\\",
      "\\hline"
    )
    for (i in 1:nrow(summary_data)) {
      lines <- c(lines, sprintf("%d & %s & %s & %.4f \\\\",
                                summary_data$N[i],
                                summary_data$Test_Type[i],
                                summary_data$Test[i],
                                summary_data$KS_pvalue[i]))
    }
    lines <- c(lines, "\\hline", "\\end{tabular}")
    return(paste(lines, collapse = "\n"))
  }

  return(summary_data)
}
