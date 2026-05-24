
#' Theoretical Degree Distribution Check
#'
#' Computes theoretical degree distributions for CCMnet.
#'
#' @param fit CCM_fit object
#' @param n_sim Number of theoretical samples
#'
#' @return Updated fit object with theoretical degree distribution
#' @noRd

CCM_theoretical_check_degree <- function(fit,
                                        n_sim) {

  
  settings <- .get_distr_settings(fit$prob_distr[[1]])
  
  # Provide 'population' as a context variable in the dots
  simulated <- settings$sampler(
    p = fit$prob_distr_params[[1]], 
    n = n_sim, 
    population = fit$population,
    max_val = fit$population
  )
  
  # Convert to data.frame and apply naming logic
  df <- as.data.frame(simulated)
  colnames(df) <- paste0("deg", 0:(ncol(df)-1))
  
  fit$target_distr <- list(
    target_stats = df,
    type = "Degree"
  )
  
  return(fit)
}