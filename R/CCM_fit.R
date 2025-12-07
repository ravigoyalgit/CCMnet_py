#' CCM_fit: Run MCMC via CCMnetpy and create CCM_fit object
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
