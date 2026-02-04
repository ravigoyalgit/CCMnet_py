
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
  
  if (fit$prob_distr[[1]] == "Multinomial_Poisson") {
    fit$population <- ncol(fit$mcmc_stats)
    theoretical_pmf <- fit$prob_distr_params[[2]]
    degrees.df <- t(rmultinom(n_sim, fit$population, prob = theoretical_pmf))
  } else if (fit$prob_distr[[1]] == "DirMult") {
    degrees.df = matrix(NA, nrow = n_sim, ncol = length(fit$prob_distr_params[[1]][[1]]))
    for (i in c(1:n_sim)) {
      degrees.df[i,] <- rmultinom(1, fit$population, prob = rdirichlet(1, alpha = fit$prob_distr_params[[1]][[1]]))
    }
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
  colnames(df) <- paste0("deg", 0:(ncol(df)-1))
  
  fit$theoretical <- list(
    theory_stats = df,
    type = "Degree"
  )
  
  return(fit)
}