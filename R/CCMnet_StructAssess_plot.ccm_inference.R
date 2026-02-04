#' Plot CCM Inference Results
#'
#' @param x A \code{ccm_inference} object.
#' @param type Type of plot: \code{"test"} or \code{"network"}.
#' @param term Network term to plot.
#' @param ... Additional arguments (unused).
#'
#' @noRd
plot.ccm_inference <- function(x,
                               type = c("test", "network"),
                               term = NULL,
                               ...) {
  
  type <- match.arg(type)
  
  if (is.null(term)) {
    term <- x$alt_terms[1]
  }
  
  if (!term %in% names(x$tests)) {
    stop("No test available for term: ", term)
  }
  
  if (type == "test") {
    plot_ccm_test(x, term)
  } else {
    plot_ccm_network(x, term)
  }
}
