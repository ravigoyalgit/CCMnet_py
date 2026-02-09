#' Plot MCMC Trace for sample_ccm
#'
#' \code{CCM_traceplot} produces a trace plot of the MCMC samples from
#' \code{sample_ccm}. This is used to diagnose convergence and mixing.
#'
#' @param object A \code{ccm_sample} object.
#' @param stats string. Which statistic to plot.
#' @param ... Additional arguments passed to \code{plot()}.
#'
#' @return A trace plot for the selected MCMC chain.
#'
#' @examples
#' ccm_sample <- sample_ccm(
#'   network_stats = list("edges"),
#'   prob_distr = list("poisson"),
#'   prob_distr_params = list(list(350)),
#'   population = 50 
#' )
#' CCM_traceplot(ccm_sample, stats = "edges")
#'
#' @export


CCM_traceplot <- function(object, stats = NULL, ...) {

  # ---- Extract MCMC stats ----
  if (inherits(object, "ccm_sample")) {
    mcmc_stats <- object$mcmc_stats
    
  } else if (is.data.frame(object)) {
    mcmc_stats <- object
    
  } else if (is.list(object) && !is.null(object$G_stats.df)) {
    mcmc_stats <- object$G_stats.df
    
  } else {
    stop("object must be a CCM_fit, CCM_MissingInference result, or data.frame")
  }
  
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
  
  return(p)
}

