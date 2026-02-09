#' Print CCM Inference Object
#'
#' @param x A \code{ccm_inference} object.
#' @param ... Unused.
#'
#' @noRd
print.ccm_inference <- function(x, ...) {
  
  # cat("CCM inference for network data\n\n")
  # cat("Null model:", deparse(x$null_model), "\n")
  # 
  # if (length(x$tests)) {
  #   cat("Tested terms:", names(x$tests), "\n\n")
  # }
  # 
  # for (t in x$tests) {
  #   cat("Term:", t$term, "\n")
  #   cat("Test statistic:", t$statistic, "\n")
  #   cat("p-value:", signif(t$p_value, 3), "\n\n")
  # }
  # 
  # invisible(x)
  return(NULL)
}
