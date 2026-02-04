#' Plot CCM Test Statistic
#'
#' @noRd
plot_ccm_test <- function(x, term) {
  
  test <- x$tests[[term]]
  
  df <- data.frame(T = test$T_null)
  
  ggplot2::ggplot(df, ggplot2::aes(x = T)) +
    ggplot2::geom_density() +
    ggplot2::geom_vline(xintercept = test$T_obs, linetype = "dashed") +
    ggplot2::labs(
      title = paste("Null distribution of", term, "test statistic"),
      subtitle = paste("p-value =", signif(test$p_value, 3)),
      x = test$statistic
    ) +
    ggplot2::theme_minimal()
}
