
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
  
  # if (fit$prob_distr[[1]] == "dirmult") {
  #   degrees.df = matrix(NA, nrow = n_sim, ncol = length(fit$prob_distr_params[[1]][[1]]))
  #   for (i in c(1:n_sim)) {
  #     degrees.df[i,] <- rmultinom(1, fit$population, prob = rdirichlet(1, alpha = fit$prob_distr_params[[1]][[1]]))
  #   }
  # } 
  # 
  # # Convert to data.frame and store in fit$theoretical
  # df <- as.data.frame(degrees.df)
  # colnames(df) <- paste0("deg", 0:(ncol(df)-1))
  # 
  # fit$theoretical <- list(
  #   theory_stats = df,
  #   type = "Degree"
  # )
  # 
  # return(fit)
  
  settings <- .get_distr_settings(fit$prob_distr[[1]])
  
  # Provide 'population' as a context variable in the dots
  simulated <- settings$sampler(
    p = fit$prob_distr_params[[1]], 
    n = n_sim, 
    population = fit$population
  )
  
  # Convert to data.frame and apply naming logic
  df <- as.data.frame(simulated)
  colnames(df) <- paste0("deg", 0:(ncol(df)-1))
  
  fit$theoretical <- list(
    theory_stats = df,
    type = "Degree"
  )
  
  return(fit)
}