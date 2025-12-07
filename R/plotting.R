#------------------------------------------------
# plot.CCM_fit
#------------------------------------------------
#' Plot CCM_fit object
#'
#' @param fit CCM_fit object
#' @param stats Character vector of statistics to plot, e.g., "edges" or c("deg0","deg1")
#' @param type "hist" or "density"
#' @param include_theoretical Logical. If TRUE, include theoretical distribution in the plot
#' @return ggplot object
#' @export
plot.CCM_fit <- function(fit,
                         stats = NULL,
                         type = c("density", "hist"),
                         include_theoretical = FALSE) {
  
  type <- match.arg(type)
  
  # Default: plot all columns if stats is NULL
  if (is.null(stats)) {
    stats <- colnames(fit$mcmc_stats)
  }
  
  # Prepare MCMC data
  df_mcmc <- fit$mcmc_stats %>%
    as.data.frame() %>%
    pivot_longer(cols = everything(), names_to = "stat", values_to = "count") %>%
    mutate(source = "MCMC") %>%
    filter(stat %in% stats)
  
  df_plot <- df_mcmc
  
  # Add theoretical distribution if requested
  if (include_theoretical) {
    if (is.null(fit$theoretical) || is.null(fit$theoretical$theory_stats)) {
      warning("No theoretical distribution available in fit object. Skipping.")
    } else {
      df_theory <- fit$theoretical$theory_stats %>%
        as.data.frame() %>%
        pivot_longer(cols = everything(), names_to = "stat", values_to = "count") %>%
        mutate(source = "Theoretical") %>%
        filter(stat %in% stats)
      
      df_plot <- bind_rows(df_mcmc, df_theory)
    }
  }
  
  # Plot
  p <- ggplot(df_plot, aes(x = count, color = source, fill = source)) +
    theme_bw() +
    facet_wrap(~stat, scales = "free") +
    labs(x = "Count", y = ifelse(type=="density", "Density", "Frequency"),
         title = "CCM_fit: MCMC vs Theoretical") +
    theme(legend.position = "top")
  
  if (type == "hist") {
    p <- p + geom_histogram(alpha = 0.5, position = "identity", bins = 30)
  } else if (type == "density") {
    p <- p + geom_density(alpha = 0.25)
  }
  
  return(p)
}
