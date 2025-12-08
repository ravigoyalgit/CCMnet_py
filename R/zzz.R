# Imports for functions used in your package
#' @importFrom stats rpois rmultinom sd dnbinom
#' @importFrom utils head
#' @importFrom dplyr %>% mutate filter everything all_of bind_rows
#' @importFrom tidyr pivot_longer
#' @importFrom ggplot2 ggplot aes geom_density geom_histogram facet_wrap labs theme theme_bw
#' @importFrom igraph graph_from_data_frame
#' @importFrom tibble tibble as_tibble

# Declare global variables to avoid R CMD check NOTES
if (getRversion() >= "2.15.1") {
  utils::globalVariables(
    c(
      "Prob_Distr",
      "Prob_Distr_Params",
      "count",
      "iter",
      "stat",
      "value"
    )
  )
}

