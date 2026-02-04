#' This function serves as a wrapper to perform MCMC-based network simulation 
#' under specific constraints, supporting both uni-modal and bi-modal populations.
#'
#' @param Network_stats A character vector specifying the network statistics to be used.
#' @param Prob_Distr A character string specifying the probability distribution.
#' @param Prob_Distr_Params A list or vector of parameters for the chosen \code{Prob_Distr}.
#' @param samplesize Integer. The number of network samples to draw from the MCMC. Default is 5000.
#' @param burnin Integer. The number of initial MCMC steps to discard. Default is 1000.
#' @param interval Integer. The number of steps between successive samples (thinning). Default is 1000.
#' @param statsonly Logical. If \code{TRUE}, returns only the network statistics; 
#' if \code{FALSE}, returns the network objects. Default is \code{TRUE}.
#' @param G An optional initial graph object.
#' @param P An optional graph object or matrix representing the population or constraints.
#' @param population Integer or vector of length 2. The size of the population. 
#' A single value triggers \code{uni_modal_constr}, while two values trigger \code{bi_modal_constr}.
#' @param covPattern A vector or data frame containing nodal attributes/covariates.
#' @param bayesian_inference Logical. Whether to perform Bayesian inference. Default is \code{FALSE}.
#' @param Ia,Il,R Optional parameters for epidemiological model states (Infectious asymptomatic, 
#' Infectious latent, Recovered).
#' @param epi_params A list of parameters for epidemiological simulations.
#' @param print_calculations Logical. If \code{TRUE}, prints progress and intermediate steps to the console.
#' @param use_G Logical. Whether to use the provided graph \code{G} as the starting state.
#' @param outfile Character string. Path to a file where results should be saved.
#' @param partial_network Logical. Whether the input network is partially observed.
#' @param obs_nodes A vector of indices for observed nodes.
#' @param MH_proposal_type Character. The Metropolis-Hastings proposal mechanism (e.g., "TNT" for Tie-No-Tie).
#' @param Obs_stats The observed statistics to match or use as constraints.
#' @param remove_var_last_entry Logical. Internal flag for variance calculation adjustments.
#'
#' @return Depending on \code{statsonly}, returns either a matrix of network statistics 
#' or a list of network objects produced by the underlying \code{uni_modal_constr} 
#' or \code{bi_modal_constr} functions.
#'
#' @noRd

CCMnet_constr <- function(Network_stats, 
                          Prob_Distr, 
                          Prob_Distr_Params,
                          samplesize = 5000, 
                          burnin=1000, 
                          interval=1000,
                          statsonly=TRUE,
                          G=NULL,
                          P=NULL,
                          population, 
                          covPattern = NULL,
                          bayesian_inference = FALSE,
                          Ia = NULL, 
                          Il = NULL, 
                          R = NULL, 
                          epi_params = NULL,
                          print_calculations = FALSE,
                          use_G = FALSE,
                          outfile = NULL,
                          partial_network = FALSE,
                          obs_nodes = NULL,
                          MH_proposal_type = "TNT",
                          Obs_stats = Obs_stats,
                          remove_var_last_entry = FALSE) {

  if (length(population) == 1) {
    return(uni_modal_constr(Network_stats = Network_stats, 
                            Prob_Distr = Prob_Distr, 
                            Prob_Distr_Params = Prob_Distr_Params,
                            samplesize = samplesize, 
                            burnin = burnin, 
                            interval = interval,
                            statsonly = statsonly, 
                            G = G,
                            population = population, 
                            covPattern = covPattern, 
                            remove_var_last_entry = remove_var_last_entry,
                            Obs_stats = Obs_stats)
           )
  } else if (length(population) == 2) {
    return(bi_modal_constr(Network_stats, Prob_Distr, Prob_Distr_Params,
                           samplesize, burnin, interval,
                           statsonly, G,
                           population, covPattern, remove_var_last_entry,
                           Obs_stats)
           )
  }
}

