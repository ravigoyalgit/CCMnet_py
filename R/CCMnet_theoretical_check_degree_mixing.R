
#' Theoretical Degree Distribution + Mixing Check
#'
#' Computes theoretical degree distributions for CCMnet.
#'
#' @param fit CCM_fit object
#' @param n_sim Number of theoretical samples
#'
#' @return Updated fit object with theoretical degree + mixing distribution
#' @noRd

CCM_theoretical_check_degree_mixing <- function(fit,
                                        n_sim) {
  
  if (fit$prob_distr[[1]] == "Normal") {
    degrees_0.df <- rmvnorm(n = n_sim, 
            mean = fit$prob_distr_params[[1]][[1]][[1]],
            sigma = fit$prob_distr_params[[1]][[2]][[1]])
    degrees_1.df <- rmvnorm(n = n_sim, 
                            mean = fit$prob_distr_params[[1]][[1]][[2]],
                            sigma = fit$prob_distr_params[[1]][[2]][[2]])
    mixing.df = rnorm(n = n_sim, 
                      mean = fit$prob_distr_params[[2]][[1]],
                      sd = sqrt(fit$prob_distr_params[[2]][[2]]))
    
    degrees_mixing.df = bind_cols(bind_cols(degrees_0.df, degrees_1.df),
                                  mixing.df)
    
    df <- as.data.frame(degrees_mixing.df)
    len_deg = length(fit$prob_distr_params[[1]][[1]][[1]])
    cov0_names = paste(paste0("deg", 0:(len_deg-1)), "_1", sep = "")
    cov1_names = paste(paste0("deg", 0:(len_deg-1)), "_2", sep = "")
    mix_names = c("M21")
    colnames(df) <- c(cov0_names, cov1_names, mix_names)

  } else {
    warning("Theoretical distribution not currently implemented. Returning NULL.")
    fit$theoretical <- list(
      theory_stats = NULL,
      type = "Degree_Mixing"
    )
    return(fit)
  }
  
  # Convert to data.frame and store in fit$theoretical

  
  fit$theoretical <- list(
    theory_stats = df,
    type = "Degree_Mixing"
  )
  
  return(fit)
}