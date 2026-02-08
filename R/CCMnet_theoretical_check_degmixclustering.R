#' Theoretical Mixing Distribution Check
#'
#' Computes theoretical mixing distributions for CCMnet.
#'
#' @param fit CCM_fit object
#' @param n_sim Number of theoretical samples
#'
#' @return Updated fit object with theoretical mixing distribution
#' @noRd

CCM_theoretical_check_degmixclustering <- function(fit,
                                         n_sim) {
  
  if (fit$prob_distr[[1]] == "Multinomial_Poisson" && fit$prob_distr[[2]] == "Normal") {
    
    lambda <- fit$prob_distr_params[[1]][[1]][1]
    probs  <- fit$prob_distr_params[[1]][[2]]
    
    # Simulate Poisson-Multinomial draws
    simulated_1 <- matrix(NA, nrow = n_sim, ncol = length(probs))
    
    for (i in seq_len(n_sim)) {
      total_edges <- rpois(1, lambda)
      simulated_1[i, ] <- rmultinom(1, size = total_edges, prob = probs)
    }
  } else if (fit$prob_distr[[1]] == "Multivariate_normal" && fit$prob_distr[[2]] == "Normal") {
    
    mean_vec <- fit$prob_distr_params[[1]][[1]]
    invsigma_mat  <- fit$prob_distr_params[[1]][[2]]
    sigma_mat = solve(invsigma_mat)
    
    simulated_1 <- rmvnorm(n_sim, mean = mean_vec, sigma = sigma_mat)

  } else if (fit$prob_distr[[2]] == "Normal" && fit$prob_distr[[2]] == "Normal") {
    
    mean_vec <- fit$prob_distr_params[[1]][[1]]
    sigma_mat  <- fit$prob_distr_params[[1]][[2]]

    simulated_1 <- rmvnorm(n_sim, mean = mean_vec, sigma = sigma_mat)
    
    mean_scalar <- fit$prob_distr_params[[2]][[1]]
    var_scalar  <- fit$prob_distr_params[[2]][[2]]
    
    simulated_2 <- rnorm(n_sim, mean = mean_scalar, sd = sqrt(var_scalar))
    
  } else {
    warning("Theoretical distribution not currently implemented. Returning NULL.")
    fit$theoretical <- list(
      theory_stats = NULL,
      type = "degmix"
    )
    return(fit)
  }
  
  simulated_1 <- as.data.frame(simulated_1)
  simulated_2 <- as.data.frame(simulated_2)
  
  simulated = bind_cols(simulated_1, simulated_2)
  
  m <- (-1 + sqrt(1 + 8*ncol(simulated_1)))/2
  degmix_clustering_names <- c()
  for (i in (seq_len(m))) {
    for (j in 1:(i)) {
      degmix_clustering_names <- c(degmix_clustering_names, paste0("DM", j, i))
    }
  }
  degmix_clustering_names = c(degmix_clustering_names , "triangles")
  colnames(simulated) <- degmix_clustering_names 
    
  fit$theoretical <- list(
    theory_stats = simulated,
    type = "degmix_clustering"
  )
  
  return(fit)
}