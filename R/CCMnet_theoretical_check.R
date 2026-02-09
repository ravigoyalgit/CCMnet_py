#' Compare MCMC Estimates with Theoretical Distributions
#'
#' This function compares the empirical MCMC distribution from \code{sample_ccm}
#' with the theoretical distribution from the CCM model. It is used as a diagnostic
#' tool to evaluate model adequacy.
#'
#' @param fit An object returned by \code{ccm_sample}.
#' @param n_sim The number of samples drawn from the theoretical distribution
#'
#' @return A diagnostic plot comparing empirical and theoretical distributions.
#'
#' @examples
#' ccm_sample <- sample_ccm(
#'   network_stats = list("edges"),
#'   prob_distr = list("poisson"),
#'   prob_distr_params = list(list(350)),
#'   population = 50 
#' )
#' ccm_sample<- CCM_theoretical_check(ccm_sample, n_sim = 1000)
#' plot(ccm_sample, stats = "edges", type = "hist", include_theoretical = TRUE)
#'
#' @export

CCM_theoretical_check <- function(
    fit,
    n_sim = nrow(fit$mcmc_stats)
) {
  stat <- fit$network_stats

  #---------------------------
  # Network Property: edges
  #---------------------------
  if (length(stat) == 1 && stat == "edges") {
    return(CCM_theoretical_check_edges(fit,
                                       n_sim))
  }
  
  #---------------------------
  # Network Property: Density
  #---------------------------
  if (length(stat) == 1 && stat == "Density") {
    return(CCM_theoretical_check_density(fit,
                                       n_sim))
  }
  
  #---------------------------
  # Network Property: Mixing
  #---------------------------
  if (length(stat) == 1 && stat == "Mixing") {
    return(CCM_theoretical_check_mixing(fit,
                                       n_sim))
  }
  
  #---------------------------
  # Network Property: Degree
  #---------------------------
  if (length(stat) == 1 && (stat == "Degree" || stat == "DegreeDist")) {
    return(CCM_theoretical_check_degree(fit,
                                        n_sim))
  }
  
  if ((length(stat) == 1 && stat == "degmix") || (length(stat) == 1 && stat == "DegMixing"))  {
    return(CCM_theoretical_check_degmix(fit,
                                        n_sim))
  }
  
  if ((length(stat) == 1 && stat == "degmix_clustering") || (length(stat) == 2 && stat[1] == "DegMixing" && stat[2] == "Triangles")) {
    return(CCM_theoretical_check_degmixclustering(fit,
                                                  n_sim))
  }
  
  if ((length(stat) == 2 && stat[1] == "DegreeDist" && stat[2] == "Mixing")) {
    return(CCM_theoretical_check_degree_mixing(fit,
                                                  n_sim))
  }
  
  stop("Theoretical distribution not implemented for this statistic.")
}
