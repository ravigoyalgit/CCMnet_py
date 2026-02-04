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
  
  if (fit$prob_distr[[1]] == "Beta") {
    density <- rbeta(n_sim, shape1 = fit$prob_distr_params[[1]][[1]][1], shape2 = fit$prob_distr_params[[1]][[2]][1]) 
  } else if (fit$prob_distr[[1]] == "Normal") {
    density <- rnorm(n_sim, mean = fit$prob_distr_params[[1]][[1]][1], sd = sqrt(fit$prob_distr_params[[1]][[2]][1])) 
  } else {
    warning("Theoretical distribution not currently implemented. Returning NULL.")
    fit$theoretical <- list(
      theory_stats = NULL,
      type = "Density"
    )
    return(fit)
  }
  
  df <- data.frame(density = density)
  
  fit$theoretical <- list(
    theory_stats = df,
    type = "Density"
  )
  
  return(fit)
}