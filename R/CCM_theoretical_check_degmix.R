#' Theoretical Mixing Distribution Check
#'
#' Computes theoretical mixing distributions for CCMnet.
#'
#' @param fit CCM_fit object
#' @param n_sim Number of theoretical samples
#'
#' @return Updated fit object with theoretical mixing distribution
#' @keywords internal

CCM_theoretical_check_degmix <- function(fit,
                                         n_sim) {
  
  if (fit$Prob_Distr[[1]] == "Multinomial_Poisson") {
    
    lambda <- fit$Prob_Distr_Params[[1]][1]
    probs  <- fit$Prob_Distr_Params[[2]]
    
    # Simulate Poisson-Multinomial draws
    simulated <- matrix(NA, nrow = n_sim, ncol = length(probs))
    
    for (i in seq_len(n_sim)) {
      total_edges <- rpois(1, lambda)
      simulated[i, ] <- rmultinom(1, size = total_edges, prob = probs)
    }
  } else if (fit$Prob_Distr[[1]] == "Multivariate_normal") {
    
    mean_vec <- fit$Prob_Distr_Params[[1]]
    invsigma_mat  <- fit$Prob_Distr_Params[[2]]
    sigma_mat = solve(invsigma_mat)
    
    simulated <- rmvnorm(n_sim, mean = mean_vec, sigma = sigma_mat)

  } else {
    warning("Theoretical distribution not currently implemented. Returning NULL.")
    fit$theoretical <- list(
      theory_stats = NULL,
      type = "degmix"
    )
    return(fit)
  }
  
  simulated <- as.data.frame(simulated)

  m <- fit$population - 1
  degmix_names <- c()
  for (i in (seq_len(m))) {
    for (j in i:(m)) {
      degmix_names <- c(degmix_names, paste0("DM", j, i))
    }
  }
  colnames(simulated) <- degmix_names 
    
  fit$theoretical <- list(
    theory_stats = simulated,
    type = "degmix"
  )
  
  return(fit)
}