#' Construct a Congruence Class Model (CCM) Network Using CCMnet (R wrapper for Python)
#'
#' @description
#' CCMnet_constr() is an R wrapper around the underlying Python function
#' CCMnet_constr_py, which generates constrained networks or sufficient
#' statistics used by CCM_fit(). This function handles argument
#' conversion, calls the Python engine, and returns an igraph object and
#' associated constraint information.
#'
#' @param Network_stats Character vector of statistic names.
#' @param Prob_Distr Character vector of probability distribution names.
#' @param Prob_Distr_Params List of parameter sets for each distribution.
#' @param samplesize Integer. Number of MCMC samples.
#' @param burnin Integer. Burn-in period for MCMC.
#' @param interval Integer. Thinning interval.
#' @param statsonly Logical. If TRUE, only return sufficient statistics.
#' @param G Initial network or starting graph object.
#' @param P Additional parameters used in the Python backend.
#' @param population Integer. Number of nodes.
#' @param covPattern Integer vector. Covariate pattern or group labels.
#' @param bayesian_inference Logical. Whether to use Bayesian inference mode.
#' @param Ia,Il,R Numeric vectors for epidemic parameters.
#' @param epi_params Additional epidemic model parameters.
#' @param print_calculations Logical. Print internal progress from Python engine.
#' @param use_G Logical. Whether to use the supplied initial graph G.
#' @param outfile Character. Path for logging output.
#' @param partial_network Numeric. Fraction of nodes observed.
#' @param obs_nodes Vector of observed node IDs.
#' @param MH_proposal_type Character. MCMC proposal type (e.g., "random").
#'
#' @return A list with two elements:
#' \itemize{
#'   \item g: An igraph network object.
#'   \item stats: Constraint or statistic output from Python.
#' }
#'
#' @examples
#' \dontrun{
#' result <- CCMnet_constr(
#'   Network_stats = list("Edge"),
#'   Prob_Distr = list("NP"),
#'   Prob_Distr_Params = list(dnbinom(0:choose(50,2), size = 1.017340, mu = 6.192894)),
#'   samplesize = 1000,
#'   burnin = 100,
#'   interval = 10,
#'   statsonly = FALSE,
#'   G = NULL,
#'   P = NULL,
#'   population = 50,
#'   covPattern = rep(1, 50),
#'   bayesian_inference = FALSE,
#'   Ia = NULL, Il = NULL, R = NULL,
#'   epi_params = NULL,
#'   print_calculations = FALSE,
#'   obs_nodes = NULL
#' )
#' }
#'
#' @export

CCMnet_constr <- function(Network_stats,
                          Prob_Distr,
                          Prob_Distr_Params, 
                          samplesize,
                          burnin, 
                          interval,
                          statsonly,
                          G,
                          P,
                          population, 
                          covPattern,
                          bayesian_inference,
                          Ia, 
                          Il, 
                          R, 
                          epi_params,
                          print_calculations,
                          use_G = FALSE,
                          outfile = "none",
                          partial_network=0,
                          obs_nodes,
                          MH_proposal_type= "random") {
  
  samplesize = as.integer(samplesize)
  burnin = as.integer(burnin)
  interval = as.integer(interval)
  population = as.integer(population)
  covPattern = as.integer(covPattern)
  
  if (!exists("CCMnet_constr_py")) {
    CCMnet_python_setup()
  }
  
  results = CCMnet_constr_py(Network_stats,
                   Prob_Distr,
                   Prob_Distr_Params, 
                   samplesize,
                   burnin, 
                   interval,
                   statsonly,
                   G,
                   P,
                   population, 
                   covPattern,
                   bayesian_inference,
                   Ia, 
                   Il, 
                   R, 
                   epi_params,
                   print_calculations,
                   use_G,
                   outfile,
                   partial_network,
                   obs_nodes,
                   MH_proposal_type)
  
  nodes_attr_df = data.frame(name = c(1:(population)), #data.frame(name = c(0:(population-1)), 
                             covPattern = covPattern)
  g = graph_from_data_frame(results[[1]], directed=FALSE, vertices = nodes_attr_df)

  return(list(g, results[[2]]))
}
  
  