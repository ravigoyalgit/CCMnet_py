#' Theoretical Edge Distribution Check
#'
#' Computes theoretical edge count distributions for CCMnet.
#'
#' @param fit CCM_fit object
#' @param n_sim Number of theoretical samples
#'
#' @return Updated fit object with theoretical edge distribution
#' @noRd

CCM_theoretical_check_edges <- function(fit,
                                        n_sim) {

  settings <- .get_distr_settings(fit$prob_distr[[1]])
  
  # Prepare the context (max_val, etc.)
  max_edges <- choose(fit$population, 2)
  
  # Call sampler blindly. It handles its own logic.
  draws <- settings$sampler(
    p = fit$prob_distr_params[[1]], 
    n = n_sim, 
    max_val = max_edges
  )
  
  fit$target_distr <- list(
    target_stats = data.frame(edges = as.vector(draws)),
    type = "Edge"
  )
  return(fit)
}