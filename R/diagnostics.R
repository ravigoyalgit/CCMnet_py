#' Traceplot for one or more statistic columns
#' @param fit CCM_fit object
#' @param stats statistic names or indices (default: first column)
#' @export
CCM_traceplot <- function(fit, stats = 1) {
  if (!inherits(fit, "CCM_fit")) stop("fit must be a CCM_fit object.")
  
  mcmc_stats <- fit$mcmc_stats
  
  # Convert numeric indices to names
  if (is.numeric(stats)) stats <- colnames(mcmc_stats)[stats]
  
  # Validate column names
  if (!all(stats %in% colnames(mcmc_stats))) stop("Some statistics not found in mcmc_stats")
  
  # Reshape data to long format for ggplot
  df <- mcmc_stats[, stats, drop = FALSE] %>%
    tibble::rownames_to_column("iter") %>%
    tidyr::pivot_longer(cols = all_of(stats), names_to = "stat", values_to = "value")
  
  # Plot with facets for multiple statistics
  p <- ggplot2::ggplot(df, ggplot2::aes(x = as.integer(iter), y = value)) +
    ggplot2::geom_line() +
    ggplot2::facet_wrap(~stat, scales = "free_y") +
    ggplot2::labs(title = "Traceplots", x = "Iteration", y = "Value") +
    ggplot2::theme_minimal()
  
  print(p)
  invisible(p)
}

