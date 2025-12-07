#' @export
print.CCM_fit <- function(x, ...) {
  cat("Object of class 'CCM_fit'\n")
  cat("-------------------------\n")
  if (!is.null(x$Network_stats)) {
    cat("Statistics:       ", paste(unlist(x$Network_stats), collapse=", "), "\n")
  }
  if (!is.null(x$Prob_Distr)) {
    cat("Distribution(s):  ", paste(unlist(x$Prob_Distr), collapse=", "), "\n")
  }
  if (!is.null(x$population)) {
    cat("Population:       ", x$population, "\n")
  }
  if (!is.null(x$stats)) {
    cat("MCMC samples:     ", nrow(x$stats), "rows x", ncol(x$stats), "cols\n")
  }
  invisible(x)
}

#' @export
summary.CCM_fit <- function(object, ...) {
  cat("Summary of CCM_fit object\n")
  cat("-------------------------\n")
  if (!is.null(object$stats)) {
    for (i in seq_len(ncol(object$stats))) {
      nm <- if (!is.null(colnames(object$stats))) colnames(object$stats)[i] else paste0("V", i)
      cat("\nStatistic:", if(!is.null(object$Network_stats[[i]])) object$Network_stats[[i]] else nm, "\n")
      vals <- object$stats[, i]
      print(summary(vals))
    }
  } else {
    cat("No stats available\n")
  }
  invisible(object)
}

#' @export
plot.CCM_fit <- function(x, stat_index = 1, theoretical_sim = NULL, ...) {
  if (!inherits(x, "CCM_fit")) stop("x must be a CCM_fit object")
  if (is.null(x$stats)) stop("No stats in object")
  if (is.numeric(stat_index)) {
    if (stat_index < 1 || stat_index > ncol(x$stats)) stop("stat_index out of range")
    idx <- stat_index
  } else {
    idx <- which(colnames(x$stats) == stat_index)
    if (length(idx) == 0) stop("stat_index name not found in stats columns")
    idx <- idx[1]
  }
  stat_name <- if (!is.null(x$Network_stats[[idx]])) x$Network_stats[[idx]] else colnames(x$stats)[idx]
  stat_type <- if (!is.null(x$Prob_Distr[[idx]])) x$Prob_Distr[[idx]] else NA
  
  library(ggplot2)
  library(tidyr)
  library(dplyr)
  
  if (!is.na(stat_type) && stat_type == "NP") {
    df <- data.frame(value = x$stats[, idx])
    p <- ggplot(df, aes(x = value)) +
      geom_density(fill = "steelblue", alpha = 0.4) +
      labs(title = paste("MCMC distribution:", stat_name), x = stat_name, y = "Density") +
      theme_minimal()
    if (!is.null(theoretical_sim)) {
      if (is.numeric(theoretical_sim)) {
        df2 <- data.frame(value = theoretical_sim)
        p <- p + geom_density(data = df2, aes(x = value), color = "red", fill = NA)
      } else if (is.data.frame(theoretical_sim) && stat_name %in% colnames(theoretical_sim)) {
        df2 <- data.frame(value = theoretical_sim[[stat_name]])
        p <- p + geom_density(data = df2, aes(x = value), color = "red", fill = NA)
      }
    }
    print(p)
    return(invisible(p))
  }
  
  # For multinomial-like outputs: treat columns as categories
  df_all <- as.data.frame(x$stats)
  if (is.null(colnames(df_all))) colnames(df_all) <- paste0("V", seq_len(ncol(df_all)))
  df_long <- df_all %>% mutate(iter = row_number()) %>%
    pivot_longer(cols = -iter, names_to = "category", values_to = "count")
  p2 <- ggplot(df_long, aes(x = count, color = category, fill = category)) +
    geom_density(alpha = 0.25) +
    facet_wrap(~category, scales = "free", ncol = 5) +
    labs(title = "MCMC samples by category", x = "Count", y = "Density") +
    theme_bw()
  print(p2)
  invisible(p2)
}
