
#' Theoretical Degree Distribution Check
#'
#' Computes theoretical degree distributions for CCMnet.
#'
#' @param fit CCM_fit object
#' @param n_sim Number of theoretical samples
#'
#' @return Updated fit object with theoretical degree distribution
#' @keywords internal

CCM_theoretical_check_degree <- function(fit,
                                        n_sim) {
  
  if (Prob_Distr[[1]] == "Multinomial_Poisson") {
    population <- ncol(fit$mcmc_stats)
    theoretical_pmf <- Prob_Distr_Params[[2]]
    degrees.df <- t(rmultinom(n_sim, population, prob = theoretical_pmf))
  } else {
    warning("Theoretical distribution not currently implemented. Returning NULL.")
    fit$theoretical <- list(
      theory_stats = NULL,
      type = "Degree"
    )
    return(fit)
  }
  
  # Convert to data.frame and store in fit$theoretical
  df <- as.data.frame(degrees.df)
  colnames(df) <- paste0("deg", 0:(population-1))
  
  fit$theoretical <- list(
    theory_stats = df,
    type = "Degree"
  )
  
  return(fit)
}