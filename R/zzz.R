# Imports for functions used in your package
#' @useDynLib CCMnet, .registration = TRUE
#' @importFrom stats rnorm rbeta dpois rpois rmultinom rgamma sd dnbinom terms
#' @importFrom utils head
#' @importFrom dplyr %>% mutate filter everything all_of bind_rows bind_cols
#' @importFrom tidyr pivot_longer
#' @import ggplot2
#' @importFrom tibble tibble as_tibble
#' @importFrom intergraph asIgraph
#' @importFrom kableExtra kable
#' @importFrom RBesT postmix mixbeta mixnorm
#' @importFrom gtools rdirichlet
#' @importFrom mvtnorm rmvnorm
#' @importFrom ergm ergm
#' @importFrom network network network.initialize %v%<- network.size
#' @importFrom rlang is_empty .data
#' @import igraph

# # Declare global variables to avoid R CMD check NOTES
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

.onLoad <- function(libname, pkgname) {
  # DO NOT initialize python
  # Optionally:
  # py_require_ccmnet()
}
