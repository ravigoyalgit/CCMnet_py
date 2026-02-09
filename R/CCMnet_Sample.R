#' Fit a Congruence Class Model (CCM)
#'
#' \code{sample_ccm} fits a Congruence Class Model using an MCMC framework to sample 
#' networks that match specific topological property distributions. It facilitates 
#' sampling based on specified network statistics (e.g., edges, degree distribution, 
#' mixing patterns) and their associated probability distributions.
#'
#' @param network_stats Character vector of statistic names to be constrained (e.g., "edges", "degree", "density").
#' @param prob_distr Character vector of probability distribution names corresponding to each statistic.
#' @param prob_distr_params List of parameter sets for each specified distribution.
#' @param population Integer. The number of nodes in the network.
#' @param sample_size Integer. Number of MCMC samples to return. Default is 1000.
#' @param burnin Integer. Number of MCMC iterations to discard before sampling begins. Default is 200,000.
#' @param interval Integer. Thinning interval (number of iterations between samples). Default is 1000.
#' @param cov_pattern Integer vector. Optional group labels or covariate patterns for nodes.
#' @param initial_g An \code{igraph} object. The starting graph for the MCMC chain.
#' @param use_initial_g Logical. If TRUE, the MCMC chain starts from \code{initial_g}.
#' @param partial_network Integer. Reserved for future use in partial network observation.
#' @param obs_nodes Integer vector. Reserved for future use in specifying observed nodes.
#' @param Obs_stats Character vector of additional network statistics to monitor during sampling.
#' @param remove_var_last_entry Logical. If TRUE, removes the variance constraint from the last entry of the distribution.
#' @param stats_only Logical. If TRUE, only sufficient statistics are returned; otherwise, network objects are included.
#'
#' @return An object of class \code{ccm_sample} containing:
#' \itemize{
#'   \item \code{mcmc_stats}: A data frame of sampled network statistics.
#'   \item \code{population}: The number of nodes in the network.
#'   \item \code{prob_distr}: The distributions used for constraints.
#'   \item \code{prob_distr_params}: Parameters used for the constraints.
#'   \item \code{network_stats}: The names of the statistics constrained.
#'   \item \code{cov_pattern}: The covariate pattern used.
#'   \item \code{theoretical}: Theoretical distribution values, if calculated.
#'   \item \code{g}: A list of sampled graphs.
#' }
#'
#' @examples
#' # Basic sampling of a random graph with an edge constraint
#' ccm_sample <- sample_ccm(
#'   network_stats = list("edges"),
#'   prob_distr = list("poisson"),
#'   prob_distr_params = list(list(350)),
#'   population = 50 
#' )
#' summary(ccm_sample)
#' plot(ccm_sample, stats = "edges", type = "hist")
#'
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
