#' Generate a Partial Observed Network from a Congruence Class Model (CCM)
#'
#' This function generates a "true" network using \code{CCM_fit} and returns a subgraph
#' representing a partial observed network based on a specified sampling fraction.
#'
#' @param population Integer. Total number of nodes in the network.
#' @param sample_fraction Numeric (0,1]. Fraction of nodes to include in the observed network.
#' @param Network_stats List. Network statistics to constrain (e.g., \code{list("Degree")}).
#' @param Prob_Distr List. Probability distributions for network statistics (e.g., \code{list("Multinomial_Poisson")}).
#' @param Prob_Distr_Params List. Parameters corresponding to the specified probability distributions.
#' @param covPattern Integer vector. Covariate pattern for nodes (length = \code{population}).
#' @param n_mcmc Integer. Number of MCMC samples to generate in \code{CCM_fit}. Default 1.
#' @param burnin Integer. Number of burn-in iterations for \code{CCM_fit}. Default 200000.
#' @param interval Integer. Interval between stored MCMC samples in \code{CCM_fit}. Default 10.
#' @param verbose Logical. If TRUE, prints progress messages from \code{CCM_fit}. Default FALSE.
#' @param seed Integer. Random seed for reproducibility. Default 12345.
#'
#' @return A list containing:
#' \item{g_obs}{The observed subgraph (igraph object).}
#' \item{g_truth}{The full generated network (igraph object).}
#' \item{sample_ids}{Indices of sampled nodes included in \code{g_obs}.}
#' \item{G_stats_truth}{Network statistics of the full network from \code{CCM_fit}.}
#' \item{fit}{The CCM_fit object used to generate the network.}
#' \item{population}{Number of nodes in the network.}
#' \item{sample_fraction}{Fraction of nodes included in \code{g_obs}.}
#'
#' @examples
#' \dontrun{
#' library(igraph)
#' population <- 100
#' sample_fraction <- 0.8
#' Network_stats <- list("Degree")
#' Prob_Distr <- list("Multinomial_Poisson")
#' Prob_Distr_Params <- list( # user must define appropriately
#'   1,                       # placeholder for edges or mixing param
#'   dpois(0:(population-1), lambda = 5)
#' )
#' covPattern <- rep(0, population)
#' result <- generate_partial_network(
#'   population = population,
#'   sample_fraction = sample_fraction,
#'   Network_stats = Network_stats,
#'   Prob_Distr = Prob_Distr,
#'   Prob_Distr_Params = Prob_Distr_Params,
#'   covPattern = covPattern
#' )
#' g_obs <- result$g_obs
#' g_truth <- result$g_truth
#' }
#'
#' @noRd

generate_partial_network <- function(population,
                                     sample_fraction,
                                     Network_stats,
                                     Prob_Distr,
                                     Prob_Distr_Params,
                                     covPattern,
                                     n_mcmc = 1L,
                                     burnin = 200000L,
                                     interval = 10L,
                                     seed = 12345) {
  # Set seed for reproducibility
  set.seed(seed)
  
  # Generate the full network using CCM_fit
  fit <- sample_ccm(network_stats = Network_stats,
                 prob_distr = Prob_Distr,
                 prob_distr_params = Prob_Distr_Params,
                 population = population,
                 sample_size = n_mcmc,
                 burnin = burnin,
                 interval = interval,
                 cov_pattern = covPattern)
  
  g_truth <- fit$g      # the full igraph network
  G_stats_truth <- fit$mcmc_stats   # network statistics
  
  # Sample observed nodes
  n_sample <- round(population * sample_fraction)
  sample_ids <- sample(0:(population-1), size = n_sample)
  
  # Create the observed subgraph
  g_obs <- igraph::induced_subgraph(g_truth, vids = sample_ids + 1)  # igraph is 1-based
  
  # Return results
  return(list(
    g_obs = g_obs,
    g_truth = g_truth,
    sample_ids = sample_ids,
    G_stats_truth = G_stats_truth,
    fit = fit,
    population = population,
    sample_fraction = sample_fraction
  ))
}
