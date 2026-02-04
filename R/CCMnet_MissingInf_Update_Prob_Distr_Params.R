#' Update Probability Distribution Parameters for CCM MCMC
#'
#' This function updates the probability distribution parameters for network statistics
#' in a Congruence Class Model (CCM) during MCMC sampling. It supports Bayesian updates
#' using hyperpriors for degree and mixing distributions.
#'
#' @param g An igraph object representing the current network state.
#' @param Prob_Distr_Params_hyperprior A list of hyperprior parameters for the probability distributions.
#' @param Network_stats A character vector specifying which network statistics to update (e.g., "Degree", "Mixing").
#' @param Prob_Distr A list specifying the type of probability distribution used (e.g., "Multinomial_Poisson").
#' @param Prob_Distr_Params A list of current probability distribution parameters to be updated.
#' @param G_stats A numeric vector of observed network statistics for the current network.
#' @param MCMC_wgt Numeric, weight to apply to the network statistics in the update (default = 1).
#' @param population Integer, total number of nodes in the network.
#'
#' @return An updated list of probability distribution parameters.
#'
#' @importFrom igraph degree gsize
#' @importFrom gtools rdirichlet
#' @noRd

Update_Prob_Distr_Params <- function(g,
                                     Prob_Distr_Params_hyperprior,
                                     Network_stats,
                                     Prob_Distr,
                                     Prob_Distr_Params,
                                     G_stats,
                                     MCMC_wgt,
                                     population) {
  
  # Update for Degree distribution
  if (Network_stats == "Degree") {
    if (Prob_Distr[[1]][1] == "Multinomial_Poisson") {
      if (Prob_Distr_Params_hyperprior[[1]][1] == "Dirichlet_Gamma") {
        
        alpha <- Prob_Distr_Params_hyperprior[[3]]
        
        v_deg <- igraph::degree(g, mode = "all")
        g_deghist <- (tabulate(v_deg, nbins = population)) * MCMC_wgt
        
        Prob_Distr_Params[[2]] <- as.numeric(gtools::rdirichlet(n = 1, alpha = alpha + g_deghist))
        #min_val <- min(Prob_Distr_Params[[2]][Prob_Distr_Params[[2]] > 0])
        Prob_Distr_Params[[2]][Prob_Distr_Params[[2]] == 0] <- .Machine$double.eps
        Prob_Distr_Params[[2]] <- Prob_Distr_Params[[2]] / sum(Prob_Distr_Params[[2]])
      }
    }
  }
  
  # Update for Mixing distribution
  if (Network_stats == "Mixing") {
    if (Prob_Distr[[1]][1] == "Multinomial_Poisson") {
      if (Prob_Distr_Params_hyperprior[[1]][1] == "Dirichlet_Gamma") {
        
        gamma_kappa <- Prob_Distr_Params_hyperprior[[2]][1]
        gamma_theta <- Prob_Distr_Params_hyperprior[[2]][2]
        alpha <- Prob_Distr_Params_hyperprior[[3]]
        
        g_edgecount <- igraph::gsize(g)
        
        gamma_kappa_update <- gamma_kappa + g_edgecount
        gamma_theta_update <- gamma_theta / (1 * gamma_theta + 1)
        alpha_update <- alpha + (G_stats * MCMC_wgt)
        
        Prob_Distr_Params[[1]][1] <- rgamma(1, shape = gamma_kappa_update, scale = gamma_theta_update)
        Prob_Distr_Params[[2]] <- as.numeric(gtools::rdirichlet(n = 1, alpha = alpha_update))
        min_val <- min(Prob_Distr_Params[[2]][Prob_Distr_Params[[2]] > 0]) / 1e5
        Prob_Distr_Params[[2]][Prob_Distr_Params[[2]] == 0] <- min_val
        Prob_Distr_Params[[2]] <- Prob_Distr_Params[[2]] / sum(Prob_Distr_Params[[2]])
      }
    }
  }
  
  return(Prob_Distr_Params)
}
