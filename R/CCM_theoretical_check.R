#' Compare MCMC Estimates with Theoretical Distributions
#'
#' This function compares the empirical MCMC distribution from \code{CCM_fit}
#' with the theoretical distribution from the CCM model. It is used as a diagnostic
#' tool to evaluate model adequacy.
#'
#' @param fit An object returned by \code{CCM_fit}.
#' @param n_sim The number of samples drawn from the theoretical distribution
#'
#' @return A diagnostic plot comparing empirical and theoretical distributions.
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
#' CCM_theoretical_check(fit, n_sim = nrow(fit$mcmc_stats))
#'
#' @export

CCM_theoretical_check <- function(
    fit,
    n_sim = nrow(fit$mcmc_stats)
) {
  stat <- fit$Network_stats[[1]]
  
  #---------------------------
  # Network Property: Edge
  #---------------------------
  if (stat == "Edge") {
    return(CCM_theoretical_check_edges(fit,
                                       n_sim))
  }
  
  #---------------------------
  # Network Property: Mixing
  #---------------------------
  if (stat == "Mixing") {
    return(CCM_theoretical_check_mixing(fit,
                                       n_sim))
  }
  
  #---------------------------
  # Network Property: Degree
  #---------------------------
  if (stat == "Degree") {
    return(CCM_theoretical_check_degree(fit,
                                        n_sim))
  }
  
  if (stat == "degmix") {
    return(CCM_theoretical_check_degmix(fit,
                                        n_sim))
  }
  
  if (stat == "degmix_clustering") {
    return(CCM_theoretical_check_degmixclustering(fit,
                                                  n_sim))
  }
  
  stop("Theoretical distribution not implemented for this statistic.")
}
