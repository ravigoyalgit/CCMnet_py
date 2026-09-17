#' Run MCMC for missing network inference using the Congruence Class Model (CCM)
#'
#' This function performs MCMC to estimate missing edges in a partially observed network,
#' updating the probability distribution parameters over iterations. It uses \code{CCM_fit}
#' with a single-step MCMC for each iteration.
#'
#' @param g_obs An igraph object representing the partially observed network.
#' @param sample_ids Integer vector of observed node IDs.
#' @param population Integer, number of nodes in the full network.
#' @param Network_stats List of network statistics to model (e.g., \code{list("Degree")}).
#' @param Prob_Distr List of probability distribution names corresponding to \code{Network_stats}.
#' @param Prob_Distr_Params List of current probability distribution parameters.
#' @param Prob_Distr_Params_prior List of hyperprior parameters for updating \code{Prob_Distr_Params}.
#' @param covPattern Integer vector of covariates used in the network (required by CCM_fit).
#' @param n_mcmc_trials Integer, number of MCMC iterations.
#' @param MCMC_wgt Numeric, weighting factor for MCMC updates (default 1).
#'
#' @return A list containing:
#' \item{G_stats.df}{Matrix of network statistics over MCMC iterations.}
#' \item{Prob_Distr_Params}{Final probability distribution parameters.}
#' \item{Prob_Distr_Params_prior}{Prior hyperparameters.}
#' \item{n_mcmc_trials}{Number of MCMC iterations.}
#' \item{Prob_Distr_Params.df}{Matrix of probability distribution parameters over iterations.}
#' \item{g_final}{igraph object representing the final network from MCMC.}
#' \item{sample_ids}{Observed node IDs.}
#' \item{population}{Number of nodes in the network.}
#' @noRd

CCM_MissingInference <- function(g_obs,
                                 sample_ids,
                                 population,
                                 Network_stats,
                                 Prob_Distr,
                                 Prob_Distr_Params,
                                 Prob_Distr_Params_prior,
                                 covPattern,
                                 n_mcmc_trials,
                                 MCMC_wgt = 1) {
  
  
  Prob_Distr_Params_prior = list(
    "Dirichlet_Gamma",
    c(1, population),
    rep(1/(population*10000), population)
  )
  
  Prob_Distr_Params <- list(
    1, # placeholder for edges or mixing parameter
    dnbinom(0:(population-1), size = 1.5, mu = 6)
  )
  
  # Initialize
  g <- g_obs
  G_stats.df <- NULL
  Prob_Distr_Params.df <- matrix(nrow = 0, ncol = length(Prob_Distr_Params[[2]]))
  
  # Compute initial network statistics
  G_stats <- tabulate(igraph::degree(g) + 1, nbins = population) ####This needs to be generalized
  
  Prob_Distr_Params <- Update_Prob_Distr_Params(
    g = g,
    Prob_Distr_Params_hyperprior = Prob_Distr_Params_prior,
    Network_stats = Network_stats,
    Prob_Distr = Prob_Distr,
    Prob_Distr_Params = Prob_Distr_Params,
    G_stats = G_stats,
    MCMC_wgt = MCMC_wgt,
    population = population
  )
  
  # MCMC loop
  for (mcmc_counter in seq_len(n_mcmc_trials)) {
    
    # Convert current network to edge list
    G_df <- igraph::as_data_frame(g, what = "edges")
    G_df[,1] <- as.integer(G_df[,1])
    G_df[,2] <- as.integer(G_df[,2])
    
    # Single-step CCM_fit using current network
    fit <- sample_ccm(
      network_stats = Network_stats,
      prob_distr = Prob_Distr,
      prob_distr_params = Prob_Distr_Params,
      population = population,
      cov_pattern = covPattern,
      use_initial_g = TRUE,
      initial_g = G_df,
      obs_nodes = sample_ids,
      burnin = 10000,
      sample_size = 1,
      interval = 1,
      partial_network = 1
    )
    
    # Extract updated network and statistics
    g <- fit$g  # assuming CCM_fit returns 'network' element
    
    G_stats <- fit$mcmc_stats
    
    if (is.numeric(G_stats)) {
      G_stats <- as.data.frame(t(G_stats))
    }
    
    if (is.null(G_stats.df)) {
      G_stats.df <- G_stats
    } else {
      G_stats.df <- rbind(G_stats.df, G_stats)
    }
    
    # Update probability distribution parameters
    Prob_Distr_Params <- Update_Prob_Distr_Params(
      g = g,
      Prob_Distr_Params_hyperprior = Prob_Distr_Params_prior,
      Network_stats = Network_stats,
      Prob_Distr = Prob_Distr,
      Prob_Distr_Params = Prob_Distr_Params,
      G_stats = G_stats,
      MCMC_wgt = MCMC_wgt,
      population = population
    )
    
    Prob_Distr_Params.df <- rbind(Prob_Distr_Params.df, Prob_Distr_Params[[2]])
    
  }
  
  # Return results
  return(list(
    G_stats.df = G_stats.df,
    Prob_Distr_Params = Prob_Distr_Params,
    Prob_Distr_Params_prior = Prob_Distr_Params_prior,
    n_mcmc_trials = n_mcmc_trials,
    Prob_Distr_Params.df = Prob_Distr_Params.df,
    g_final = g,
    sample_ids = sample_ids,
    population = population
  ))
}
