#' @export
print.ccmnet <- function(x, ...) {
  cat("Object of class 'ccmnet'\n")
  cat("Call:\n")
  print(x$call)
  cat("\nModel spec:\n")
  print(x$model_spec[c("Network_stats", "Prob_Distr")])
  cat("\nResult summary:\n")
  if (!is.null(x$stats)) {
    cat("  MCMC samples stored: ", nrow(x$stats), "\n")
    cat("  Statistic columns: ", paste(colnames(x$stats), collapse = ", "), "\n")
  } else {
    cat("  No stats available\n")
  }
  invisible(x)
}

#' @export
summary.ccmnet <- function(object, ...) {
  stats <- object$stats
  cat("Summary for 'ccmnet' object\n")
  cat("Timestamp:", as.character(object$timestamp), "\n")
  if (!is.null(stats)) {
    cat("\nColumn means (first 10 cols):\n")
    print(head(colMeans(stats), 10))
    cat("\nColumn sds (first 10 cols):\n")
    print(head(apply(stats, 2, sd), 10))
  } else {
    cat("No stats to summarize\n")
  }
  invisible(object)
}
