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
  
  if (fit$prob_distr[[1]] == "np") {
    pmf <- fit$prob_distr_params[[1]][[1]]
    max_edges <- choose(fit$population, 2)
    edges <- sample(0:max_edges, size = n_sim, replace = TRUE, prob = pmf)
  } else if (fit$prob_distr[[1]] == "poisson") {
    lambda <- fit$prob_distr_params[[1]][[1]]
    max_edges <- choose(fit$population, 2)
    edges <- sample(0:max_edges, size = n_sim, replace = TRUE, prob = dpois(c(0:max_edges),lambda = lambda))
  } else if (fit$prob_distr[[1]] == "uniform") {
    max_edges <- choose(fit$population, 2)
    edges <- sample(0:max_edges, size = n_sim, replace = TRUE, prob = rep(1/max_edges, max_edges+1))
  } else if (fit$prob_distr[[1]] == "normal") {
    edges <- rnorm(n_sim, mean = fit$prob_distr_params[[1]][[1]][1], sd = sqrt(fit$prob_distr_params[[1]][[2]][1])) 
  } else {
    warning("Theoretical distribution not currently implemented. Returning NULL.")
    fit$theoretical <- list(
      theory_stats = NULL,
      type = "Edge"
    )
    return(fit)
  }
  
  df <- data.frame(edges = edges)
  
  fit$theoretical <- list(
    theory_stats = df,
    type = "Edge"
  )
  
  return(fit)
}