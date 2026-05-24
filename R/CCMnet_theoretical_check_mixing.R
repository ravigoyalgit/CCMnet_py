#' Theoretical Mixing Distribution Check
#'
#' Computes theoretical mixing distributions for CCMnet.
#'
#' @param fit CCM_fit object
#' @param n_sim Number of theoretical samples
#'
#' @return Updated fit object with theoretical mixing distribution
#' 
#' @importFrom stats na.omit
#' @noRd

CCM_theoretical_check_mixing <- function(fit,
                                         n_sim) {

  settings <- .get_distr_settings(fit$prob_distr[[1]])
  
  # Even if mixing doesn't usually use max_val, we pass it for consistency
  # so the sampler signature can be identical across all distributions.
  simulated <- as.data.frame(settings$sampler(
    p = fit$prob_distr_params[[1]], 
    n = n_sim,
    population = fit$population,
    max_val = NULL # Context provided but ignored by poisson/mvn
  ))
  
  # Column naming logic remains the same
  m <- length(unique(fit$cov_pattern))
  mixing_names <- outer(1:m, 1:m, function(i, j) ifelse(i >= j, paste0("M", i, j), NA))
  colnames(simulated) <- na.omit(as.vector(t(mixing_names)))
  
  fit$target_distr <- list(target_stats = simulated, type = "mixing")
  return(fit)
}