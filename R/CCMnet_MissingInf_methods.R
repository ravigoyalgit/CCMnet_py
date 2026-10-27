#' Plot posterior draws from CCM_MissingInference
#'
#' Uses the ccm_sample plot (\code{type = "density"}, \code{"hist"} or
#' \code{"trace"}), optionally with a reference line at known true values
#' (e.g. in a simulation study). The target-distribution overlay is not
#' available: the posterior of the network statistics given the observed data
#' is not the distribution specified by \code{prob_distr}.
#'
#' @param x A \code{ccm_missing} object.
#' @param ... Passed to \code{plot.ccm_sample} (e.g. \code{stats}, \code{type}).
#' @param truth Optional true values of the statistics: a named vector (names
#'   as in \code{colnames(x$mcmc_stats)}) or an unnamed vector in that column
#'   order. Drawn as a red line (horizontal for trace plots, vertical
#'   otherwise).
#' @param target_distr Must be \code{FALSE}.
#' @export

plot.ccm_missing <- function(x, ..., truth = NULL, target_distr = FALSE) {
  if (isTRUE(target_distr)) {
    stop("target_distr = TRUE is not available for CCM_MissingInference results: ",
         "the posterior of the network statistics is not the prob_distr target. ",
         "Use `truth` to mark known true values.")
  }
  p <- plot.ccm_sample(x, ..., target_distr = FALSE)
  if (is.null(truth)) return(p)
  
  stat_names <- colnames(x$mcmc_stats)
  truth <- unlist(truth)
  if (is.null(names(truth))) {
    if (length(truth) != length(stat_names)) {
      stop(sprintf("Unnamed `truth` must have one value per statistic (%d).", length(stat_names)))
    }
    names(truth) <- stat_names
  }
  shown <- unique(as.character(p$data$stat))          # statistics in this plot
  truth <- truth[names(truth) %in% shown]
  if (length(truth) == 0) {
    warning("None of the `truth` names match the plotted statistics.")
    return(p)
  }
  df_truth <- data.frame(stat = names(truth), truth = as.numeric(truth))
  
  is_trace <- "iter" %in% names(p$data)               # trace plots have an iteration column
  if (is_trace) {
    p + ggplot2::geom_hline(data = df_truth, ggplot2::aes(yintercept = .data$truth),
                            color = "red", linewidth = 0.8)
  } else {
    p + ggplot2::geom_vline(data = df_truth, ggplot2::aes(xintercept = .data$truth),
                            color = "red", linewidth = 0.8)
  }
}