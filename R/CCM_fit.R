#' Fit a Congruence Class Model (CCM)
#'
#' \code{CCM_fit} fits a Congruence Class Model using MCMC to match
#' observed network statistics. It takes a list of network statistics and a list
#' of probability distributions, and returns an object containing MCMC samples,
#' diagnostics, and theoretical expectations if available.
#'
#' @param Network_stats Character vector of statistic names.
#' @param Prob_Distr Character vector of probability distribution names.
#' @param Prob_Distr_Params List of parameter sets for each distribution.
#' @param samplesize Integer. Number of MCMC samples.
#' @param burnin Integer. Burn-in period for MCMC.
#' @param interval Integer. Thinning interval.
#' @param statsonly Logical. If TRUE, only return sufficient statistics.
#' @param population Integer. Number of nodes.
#' @param covPattern Integer vector. Covariate pattern or group labels.
#'
#' @return An object of class \code{CCM_fit} containing:
#' \itemize{
#'   \item \code{samples}: MCMC samples of network statistics
#'   \item \code{theoretical}: theoretical distribution if available
#'   \item \code{call}: the original function call
#' }
#'
#' @examples
#' CCMnet_python_setup()
#' population = 100L
#' fit <- CCM_fit(
#'   Network_stats = list("Edge"),
#'   Prob_Distr = list("NP"),
#'   Prob_Distr_Params = list(dnbinom(0:choose(population,2), size = 1.017340, mu = 6.192894)),
#'   population = population,
#'   samplesize = 1000L,
#'   burnin = 200000L,
#'   interval = 1000L,
#'   covPattern = rep(0L, population)  
#' )
#'
#' @export

CCM_fit <- function(
    Network_stats,
    Prob_Distr,
    Prob_Distr_Params,
    population,
    samplesize = 1000L,
    burnin = 200000L,
    interval = 1000L,
    covPattern = rep(0L, population)    
) {
  
  # Call Python backend
  out <- CCMnetpy::CCMnet_constr(
    Network_stats = Network_stats,
    Prob_Distr = Prob_Distr,
    Prob_Distr_Params = Prob_Distr_Params,
    samplesize = as.integer(samplesize),
    burnin = as.integer(burnin),
    interval = as.integer(interval),
    statsonly = TRUE,
    G = NULL,
    P = NULL,
    population = as.integer(population),
    covPattern = as.integer(covPattern),
    bayesian_inference = FALSE,
    Ia = NULL,
    Il = NULL,
    R = NULL,
    epi_params = NULL,
    print_calculations = FALSE,
    use_G = FALSE,
    outfile = "none",
    partial_network = as.integer(0),
    obs_nodes = NULL,
    MH_proposal_type = "TNT"
  )
  
  # Extract MCMC statistics
  stats <- as.data.frame(out[[2]])
  
  # Assign column names dynamically
  colnames(stats) <- unlist(lapply(Network_stats, function(s) {
    s <- tolower(s)
    
    if (s == "edge") {
      return(c(
        paste0("deg", 0:(population - 1)),
        "edges"
      ))
    }
    
    if (s == "degree") {
      return(paste0("deg", 0:(population - 1)))
    }
    
    if (s == "mixing") {
      m <- length(unique(covPattern))
      mixing_names <- c()
      for (i in seq_len(m)) {
        for (j in 1:i) {
          mixing_names <- c(mixing_names, paste0("M", i, j))
        }
      }
      return(mixing_names)
    }
    
    stop(paste("Unknown Network_stats:", s))
  }))
  
  # Create CCM_fit object
  obj <- list(
    mcmc_stats = stats,
    population = population,
    Prob_Distr = Prob_Distr,
    Prob_Distr_Params = Prob_Distr_Params,
    Network_stats = Network_stats,
    theoretical = NULL
  )
  
  class(obj) <- "CCM_fit"
  return(obj)
}
