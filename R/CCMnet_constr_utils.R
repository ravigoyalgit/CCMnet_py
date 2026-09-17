#' Generate Initial Graph for CCMnet Simulation
#'
#' This function prepares the starting graph for the CCMnet MCMC routine. 
#' If an initial graph is not provided, it generates a random Erdős-Rényi graph 
#' that satisfies a specific maximum degree constraint.
#'
#' @param G An optional \code{igraph} object. If \code{NULL}, a random graph is generated.
#' @param max_degree Integer. The maximum degree allowed for any node in the generated graph.
#' @param ER_prob Numeric. The initial probability for edge creation in the Erdős-Rényi model.
#'
#' @details 
#' If \code{G} is \code{NULL}, the function enters a loop using \code{igraph::sample_gnp}. 
#' In each iteration, it checks if the maximum degree is within \code{max_degree}. 
#' If not, it halves the \code{ER_prob} and tries again.
#' 
#' Note: This function currently relies on \code{population} and \code{covPattern} 
#' being available in the global environment or the calling scope.
#'
#' @return A list containing two elements:
#' \itemize{
#'   \item \code{P}: The processed graph (identical to the generated/provided graph).
#'   \item \code{g}: The \code{igraph} object used for the simulation.
#' }
#' 
#' @importFrom igraph sample_gnp V degree
#' @noRd

generate_initial_graph_CCMnet <- function(G, max_degree, ER_prob, covPattern, population) {
  Gen_Net_counter = 1
  G_max_degree_bool = FALSE
  if (is.null(G)) {
    while (!G_max_degree_bool) {
      g <- sample_gnp(n = population, p = ER_prob, directed = FALSE)
      V(g)$CovAttribute <- covPattern
      ER_prob = ER_prob/2
      G_max_degree_bool = max(degree(g)) <= max_degree
      Gen_Net_counter =   Gen_Net_counter + 1
    }
    P = g
  } else {
    g = G
    P = G
  }
  return(list(P, g))
}