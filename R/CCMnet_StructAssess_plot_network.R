#' Plot CCM Network Statistic
#'
#' @noRd
plot_ccm_network <- function(x, term) {
  
  if (term == "DegreeDist") {
    return(plot_ccm_degree(x))
  }
  
  stop("No network plot implemented for term: ", term)
}

#' Degree Distribution Plot for CCM
#'
#' @noRd
plot_ccm_degree <- function(x) {
  
  population <- x$fit_null$population
  
  degree_obs <- igraph::degree(x$observed_network)
  degree_counts_obs <- table(factor(degree_obs,
                                    levels = 0:(population - 1)))
  
  degree_samples <- x$fit_null$mcmc_stats[, paste0("deg", 0:(population - 1))]
  degree_mean <- colMeans(degree_samples)
  
  df <- data.frame(
    degree = 0:(population - 1),
    observed = as.numeric(degree_counts_obs),
    posterior_mean = degree_mean
  )
  
  ggplot2::ggplot(df, ggplot2::aes(x = degree)) +
    ggplot2::geom_line(ggplot2::aes(y = .data$observed, color = "Observed")) +
    ggplot2::geom_line(ggplot2::aes(y = .data$posterior_mean,
                                    color = "CCM Posterior Mean")) +
    ggplot2::labs(
      y = "Number of nodes",
      color = "",
      title = "Degree distribution: observed vs CCM"
    ) +
    ggplot2::theme_minimal()
}
