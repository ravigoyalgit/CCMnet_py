#' Compare Posterior and True Network Statistics
#'
#' Produces comparison plots between posterior distributions of
#' network statistics inferred via CCM missing-data MCMC and their
#' true values obtained from a fully observed network.
#'
#' @param result A \code{CCM_MissingInference} result object.
#' @param G_stats_truth Numeric vector of true network statistics
#'   returned by \code{generate_partial_network()}.
#' @param stats Character or integer vector specifying which statistics
#'   to plot (column names or indices of \code{G_stats.df}).
#' @param levels Optional numeric vector specifying which levels
#'   (e.g., degrees) to include. Default is NULL (use all).
#' @param fill Fill color for posterior density plots.
#' @param alpha Alpha transparency for density plots.
#'
#' @return A \code{ggplot} object.
#'
#' @examples
#' \dontrun{
#' CCM_compare_stats(
#'   result = mcmc_result,
#'   G_stats_truth = G_stats_truth,
#'   stats = paste0("Degree_", 0:9)
#' )
#' }
#'
#' @noRd

CCM_compare_stats <- function(result,
                              G_stats_truth,
                              stats = NULL,
                              levels = NULL,
                              fill = "lightblue",
                              alpha = 0.5) {
  
  if (!is.list(result) || is.null(result$G_stats.df)) {
    stop("result must be a CCM_MissingInference object containing G_stats.df.")
  }
  
  G_stats.df <- as.data.frame(result$G_stats.df)
  
  if (length(G_stats_truth) != ncol(G_stats.df)) {
    stop("Length of G_stats_truth must match number of columns in G_stats.df.")
  }
  
  # ---- Determine statistics to plot ----
  if (is.null(stats)) {
    stats <- colnames(G_stats.df)
  }
  
  if (is.numeric(stats)) {
    stats <- colnames(G_stats.df)[stats]
  }
  
  if (!all(stats %in% colnames(G_stats.df))) {
    stop("Some requested stats not found in G_stats.df.")
  }
  
  # ---- Posterior (long format) ----
  post_long <- G_stats.df[, stats, drop = FALSE] %>%
    tibble::rownames_to_column("iter") %>%
    tidyr::pivot_longer(
      cols = -iter,
      names_to = "stat",
      values_to = "value"
    ) %>%
    dplyr::mutate(value = as.numeric(value))
  
  # ---- Truth ----
  truth_df <- data.frame(
    stat = colnames(G_stats.df),
    truth = as.numeric(G_stats_truth),
    stringsAsFactors = FALSE
  ) %>%
    dplyr::filter(stat %in% stats)
  
  # ---- Optional level filtering (e.g., Degree_0–Degree_9) ----
  if (!is.null(levels)) {
    level_pattern <- paste0("(", paste(levels, collapse = "|"), ")$")
    post_long <- post_long %>% dplyr::filter(grepl(level_pattern, stat))
    truth_df  <- truth_df  %>% dplyr::filter(grepl(level_pattern, stat))
  }
  
  # ---- Plot ----
  ggplot2::ggplot(post_long, ggplot2::aes(x = value)) +
    ggplot2::geom_density(fill = fill, alpha = alpha) +
    ggplot2::geom_vline(
      data = truth_df,
      ggplot2::aes(xintercept = .data$truth),
      color = "red",
      linetype = "dashed"
    ) +
    ggplot2::facet_wrap(~stat, scales = "free") +
    ggplot2::labs(
      title = "Posterior vs True Network Statistics",
      x = "Statistic Value",
      y = "Density"
    ) +
    ggplot2::theme_bw()
}
