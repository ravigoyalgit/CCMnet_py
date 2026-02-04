#' Theoretical Mixing Distribution Check
#'
#' Computes theoretical mixing distributions for CCMnet.
#'
#' @param fit CCM_fit object
#' @param n_sim Number of theoretical samples
#'
#' @return Updated fit object with theoretical mixing distribution
#' @noRd

CCM_theoretical_check_degmix <- function(fit,
                                         n_sim) {
  
  if (fit$prob_distr[[1]] == "Multinomial_Poisson") {
    
    lambda <- fit$prob_distr_params[[1]][1]
    probs  <- fit$prob_distr_params[[2]]
    
    # Simulate Poisson-Multinomial draws
    simulated <- matrix(NA, nrow = n_sim, ncol = length(probs))
    
    for (i in seq_len(n_sim)) {
      total_edges <- rpois(1, lambda)
      simulated[i, ] <- rmultinom(1, size = total_edges, prob = probs)
    }
  } else if (fit$prob_distr[[1]] == "Multivariate_normal") {
    
    mean_vec <- fit$prob_distr_params[[1]]
    invsigma_mat  <- fit$prob_distr_params[[2]]
    sigma_mat = solve(invsigma_mat)
    
    simulated <- rmvnorm(n_sim, mean = mean_vec, sigma = sigma_mat)

  } else if (fit$prob_distr[[1]] == "Normal") {
    
    mean_vec <- fit$prob_distr_params[[1]][[1]]
    sigma_mat  <- fit$prob_distr_params[[1]][[2]]

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

  m <- (-1 + sqrt(1 + 8*ncol(simulated)))/2
  degmix_names <- c()
  for (i in (seq_len(m))) {
    for (j in 1:(i)) {
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