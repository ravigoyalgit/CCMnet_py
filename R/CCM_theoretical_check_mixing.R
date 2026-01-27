#' Theoretical Mixing Distribution Check
#'
#' Computes theoretical mixing distributions for CCMnet.
#'
#' @param fit CCM_fit object
#' @param n_sim Number of theoretical samples
#'
#' @return Updated fit object with theoretical mixing distribution
#' @keywords internal

CCM_theoretical_check_mixing <- function(fit,
                                         n_sim) {
  
  if (fit$Prob_Distr[[1]] == "Poisson") {
    lambda_vec <- fit$Prob_Distr_Params[[1]][[1]]
    simulated = NULL
    for (i in c(1:length(lambda_vec))) {
      m_edges <- rpois(n_sim, lambda_vec[i])
      simulated = bind_cols(simulated, m_edges)
    }
  } else if (fit$Prob_Distr[[1]] == "Multinomial_Poisson") {
    
    lambda <- fit$Prob_Distr_Params[[1]][1]
    probs  <- fit$Prob_Distr_Params[[2]]
    
    # Simulate Poisson-Multinomial draws
    simulated <- matrix(NA, nrow = n_sim, ncol = length(probs))
    
    for (i in seq_len(n_sim)) {
      total_edges <- rpois(1, lambda)
      simulated[i, ] <- rmultinom(1, size = total_edges, prob = probs)
    }
  } else {
    warning("Theoretical distribution not currently implemented. Returning NULL.")
    fit$theoretical <- list(
      theory_stats = NULL,
      type = "Mixing"
    )
    return(fit)
  }
  
  simulated <- as.data.frame(simulated)

  m <- length(unique(fit$covPattern))
  mixing_names <- c()
  for (i in seq_len(m)) {
    for (j in 1:i) {
      mixing_names <- c(mixing_names, paste0("M", i, j))
    }
  }
  colnames(simulated) <- mixing_names
    
  fit$theoretical <- list(
    theory_stats = simulated,
    type = "Mixing"
  )
  
  return(fit)
}