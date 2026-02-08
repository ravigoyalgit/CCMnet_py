#' @export
print.ccm_sample <- function(x, ...) {
  cat("Object of class 'ccm_sample'\n")
  cat("-------------------------\n")
  if (!is.null(x$network_stats)) {
    cat("Statistics:       ", paste(unlist(x$network_stats), collapse=", "), "\n")
  }
  if (!is.null(x$prob_distr)) {
    cat("Distribution(s):  ", paste(unlist(x$prob_distr), collapse=", "), "\n")
  }
  if (!is.null(x$population)) {
    cat("Population:       ", x$population, "\n")
  }
  if (!is.null(x$mcmc_stats)) {
    cat("MCMC samples:     ", nrow(x$mcmc_stats), "rows x", ncol(x$mcmc_stats), "cols\n")
  }
  invisible(x)
}

#' @export

summary.ccm_sample <- function(object, ...) {
  cat("Summary of ccm_sample object\n")
  cat("-------------------------\n")
  if (!is.null(object$mcmc_stats)) {
    for (i in seq_len(ncol(object$mcmc_stats))) {
      nm <- if (!is.null(colnames(object$mcmc_stats))) colnames(object$mcmc_stats)[i] else paste0("V", i)
      cat("\nStatistic: ", nm, "\n")
      vals <- object$mcmc_stats[, i]
      print(summary(vals))
    }
  } else {
    cat("No stats available\n")
  }
  invisible(object)
}

#' @export
plot.ccm_sample <- function(x,
                         stats = NULL,
                         type = c("density", "hist"),
                         include_theoretical = FALSE,
                         ...) {
  
  type <- match.arg(type)
  
  fit <- x
  
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
         title = "") +
    theme(legend.position = "bottom",
          legend.title = element_blank(), # Removes "source"
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank())
  
  if (type == "hist") {
    p <- p + geom_histogram(
      aes(y = after_stat(.data$density)), 
      alpha = 0.5, 
      position = "identity", 
      binwidth = 1,
      center = 0  
    )
  } else if (type == "density") {
    p <- p + geom_density(alpha = 0.25)
  }
  
  return(p)
}
