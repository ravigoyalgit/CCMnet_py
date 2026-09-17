#' Theoretical Density Distribution Check
#'
#' Computes theoretical density distributions for CCMnet.
#'
#' @param fit CCM_fit object
#' @param n_sim Number of theoretical samples
#'
#' @return Updated fit object with theoretical density distribution
#' @noRd

CCM_theoretical_check_density <- function(fit,
                                        n_sim) {

  settings <- .get_distr_settings(fit$prob_distr[[1]])
  
  # Call the sampler. It doesn't need population or max_val, 
  # but they are passed via ... if you use a universal caller.
  draws <- settings$sampler(
    p = fit$prob_distr_params[[1]], 
    n = n_sim,
    max_val = 1
  )
  
  fit$target_distr <- list(
    target_stats = data.frame(density = as.vector(draws)),
    type = "density"
  )
  
  return(fit)
}