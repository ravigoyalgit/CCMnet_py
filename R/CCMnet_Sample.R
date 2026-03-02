#' Sample from a Congruence Class Model (CCM)
#'
#' \code{sample_ccm} generates networks from a Congruence Class Model using a 
#' Metropolis-Hastings MCMC framework. Unlike traditional models that fit parameters 
#' to a single observed graph, CCM samples from the space of all possible networks 
#' where topological properties follow specified target probability distributions.
#'
#' \subsection{Target Distributions}{
#' The model treats network statistics as random variables following a target 
#' distribution. The following table summarizes the implemented network 
#' statistics and their compatible distributions:
#' \tabular{ll}{
#'   \strong{Network Statistic} \tab \strong{Compatible Target Distributions} \cr
#'   \code{"edges"} \tab \code{"poisson"}, \code{"uniform"}, \code{"np"} \cr
#'   \code{"density"} \tab \code{"normal"}, \code{"beta"} \cr
#'   \code{"degreedist"} \tab \code{"dirmult"} \cr
#'   \code{"degmixing"} \tab \code{"mvn"} \cr
#'   \code{"mixing"} \tab \code{"poisson"} \cr
#'   \code{c("degmixing", "triangles")} \tab \code{c("mvn", "normal")} \cr
#'   \code{c("degreedist", "mixing")} \tab \code{c("mvn", "normal")} \cr
#' }
#' }
#'
#' @param network_stats Character vector of statistic names to be targeted. 
#'   For joint targets, use vectors like \code{c("degmixing", "triangles")}.
#' @param prob_distr Character vector of probability distribution names 
#'   corresponding to each statistic.
#' @param prob_distr_params List of parameter sets for each specified distribution.
#' @param population Integer. The number of nodes in the network.
#' @param sample_size Integer. Number of MCMC samples to return. Default is 1000.
#' @param burnin Integer. Number of MCMC iterations to discard before sampling begins. Default is 200,000.
#' @param interval Integer. Thinning interval (number of iterations between samples). Default is 1000.
#' @param cov_pattern Integer vector. Optional nodal attributes (group IDs) 
#'   required for mixing or degree-mixing targets.
#' @param initial_g An \code{igraph} object. The starting graph for the MCMC chain.
#' @param use_initial_g Logical. If TRUE, the MCMC chain starts from \code{initial_g}.
#' @param partial_network Integer. Reserved for future use.
#' @param obs_nodes Integer vector. Reserved for future use in specifying observed nodes.
#' @param Obs_stats Character vector of additional network statistics to 
#'   monitor (but not target) during sampling. Reserved for future use.
#' @param remove_var_last_entry Logical. If TRUE, the last entry of the variance 
#'   matrix is dropped to ensure invertibility for certain distributions.
#' @param stats_only Logical. If TRUE, only sufficient statistics are returned; 
#'   if FALSE, the list of sampled \code{igraph} objects is included.
#'
#' @return An object of class \code{ccm_sample} containing:
#' \itemize{
#'   \item \code{mcmc_stats}: A data frame of sampled network statistics.
#'   \item \code{population}: The number of nodes in the network.
#'   \item \code{prob_distr}: The names of the target distributions used.
#'   \item \code{prob_distr_params}: The parameter values used for the target distributions.
#'   \item \code{network_stats}: The names of the network statistics targeted.
#'   \item \code{cov_pattern}: The nodal covariate pattern used (if any).
#'   \item \code{theoretical}: A list containing theoretical samples, populated by calling \code{sample_theoretical()}.
#'   \item \code{g}: A list of sampled \code{igraph} objects (last network if \code{stats_only = TRUE}).
#' }
#'
#' @details 
#' The returned \code{ccm_sample} object has associated \code{plot} and 
#' \code{sample_theoretical} methods for diagnostic and comparative analysis.
#'
#' @examples
#' # 1. Define target distributions and sample from the CCM
#' ccm_sample <- sample_ccm(
#'   network_stats = "edges",
#'   prob_distr = "poisson",
#'   prob_distr_params = list(list(350)),
#'   population = 50
#' )
#' 
#' # 2. Generate theoretical samples for the same target
#' ccm_sample <- sample_theoretical(ccm_sample)
#' 
#' # 3. Visualize MCMC samples against theoretical target
#' plot(ccm_sample, type = "hist", include_theoretical = TRUE)
#' 
#' @seealso \code{\link{sample_theoretical}}, \code{\link{plot.ccm_sample}}
#' @export

sample_ccm <- function(
    network_stats,
    prob_distr,
    prob_distr_params,
    population,
    sample_size = 1000L,
    burnin = 200000L,
    interval = 1000L,
    cov_pattern = NULL,
    initial_g = NULL,
    use_initial_g = FALSE,
    partial_network = as.integer(0),
    obs_nodes = NULL,
    Obs_stats = NULL,
    remove_var_last_entry = FALSE,
    stats_only = TRUE
) {
  
  # Perform all input checks
  .validate_sample_ccm_inputs(
    network_stats, prob_distr, prob_distr_params, population,
    sample_size, burnin, interval, cov_pattern, initial_g, use_initial_g
  )
  
  # Call C backend
  out <- CCMnet_constr(
    Network_stats = network_stats,
    Prob_Distr = prob_distr,
    Prob_Distr_Params = prob_distr_params,
    samplesize = as.integer(sample_size),
    burnin = as.integer(burnin),
    interval = as.integer(interval),
    statsonly = stats_only,
    G = initial_g,
    P = NULL,
    population = as.integer(population),
    covPattern = as.integer(cov_pattern),
    bayesian_inference = FALSE,
    Ia = NULL,
    Il = NULL,
    R = NULL,
    epi_params = NULL,
    print_calculations = FALSE,
    use_G = use_initial_g,
    outfile = "none",
    partial_network = as.integer(partial_network),
    obs_nodes = obs_nodes,
    MH_proposal_type = "TNT",
    Obs_stats = Obs_stats,
    remove_var_last_entry = remove_var_last_entry
  )
  
  # Extract MCMC statistics
  stats <- as.data.frame(out[[2]])
  
  Network_stats_comb = paste(network_stats, collapse = "_")
  
  # Assign column names dynamically
  colnames(stats) <- unlist(lapply(c(Network_stats_comb,Obs_stats), function(s) {
    s <- tolower(s)
    
    if (is.null(Obs_stats)) Obs_stats <- ""
 
    if (s == "edges") {
      if (Obs_stats == "degree") {
        return(c("edges",paste0("deg", 0:(population - 1)))) 
      } else {
        return(c("edges")) 
      }
    }
    
    if (s == "density") {
        return(c("density")) 
    }
    
    if (s == "degree") {
      return(paste0("deg", 0:(population - 1)))
    }
    
    if (s == "degreedist" && Obs_stats == "") {
      return(paste0("deg", 0:(ncol(stats) - 1)))
    }
    
    if (s == "degreedist" && Obs_stats != "") {
      return(paste0("deg", 0:(ncol(stats) - 2)))
    }
    
    if (s == "mixing") {
      m <- length(unique(cov_pattern))
      mixing_names <- c()
      for (i in seq_len(m)) {
        for (j in 1:i) {
          mixing_names <- c(mixing_names, paste0("M", i, j))
        }
      }
      return(mixing_names)
    }
    
    if (s == "degmix" ) {
      m <- (-1 + sqrt(1 + 8*ncol(stats)))/2
      degmix_names <- c()
      for (i in (seq_len(m))) {
        for (j in i:(m)) {
          degmix_names <- c(degmix_names, paste0("DM", j, i))
        }
      }
      return(degmix_names)
    }

    if (s == "degmixing") {
      m <- (-1 + sqrt(1 + 8*ncol(stats)))/2
      degmix_names <- c()
      for (i in (seq_len(m))) {
        for (j in 1:(i)) {
          degmix_names <- c(degmix_names, paste0("DM", j, i))
        }
      }
      return(degmix_names)
    }
    
    if (s == "triangles") {
      return(c("triangles")) 
    }
    
    if (s == "degmixing_triangles") {
      m <- (-1 + sqrt(1 + 8*ncol(stats)))/2
      degmix_names <- c()
      for (i in (seq_len(m))) {
        for (j in 1:(i)) {
          degmix_names <- c(degmix_names, paste0("DM", j, i))
        }
      }
      return(c(degmix_names, "triangles"))
    }
    
    if (s == "degreedist_mixing") {
      len_deg = (ncol(stats) - 3) / 2
      cov0_names = paste(paste0("deg", 0:(len_deg-1)), "_1", sep = "")
      cov1_names = paste(paste0("deg", 0:(len_deg-1)), "_2", sep = "")
      mix_names = c("M11", "M21", "M22")
      return(c(cov0_names, cov1_names, mix_names))
    }
    
    if (s == "degmix_clustering") {
      m <- population - 1
      degmix_clustering_names <- c()
      for (i in (seq_len(m))) {
        for (j in i:(m)) {
          degmix_clustering_names <- c(degmix_clustering_names, paste0("DM", j, i))
        }
      }
      degmix_clustering_names <- c(degmix_clustering_names, "triangles")
      return(degmix_clustering_names)
    }
    
    stop(paste("Unknown Network_stats:", s))
  }))
  
  # Create CCM_fit object
  obj <- list(
    mcmc_stats = stats,
    population = population,
    prob_distr = prob_distr,
    prob_distr_params = prob_distr_params,
    network_stats = network_stats,
    cov_pattern = cov_pattern,
    theoretical = NULL,
    g = out[[1]]
  )
  
  class(obj) <- "ccm_sample"
  return(obj)
}
