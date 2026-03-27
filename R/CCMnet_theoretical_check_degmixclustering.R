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
  
  # if (fit$prob_distr[[1]] == "mvn" && fit$prob_distr[[2]] == "normal") {
  #   
  #   mean_vec <- fit$prob_distr_params[[1]][[1]]
  #   sigma_mat  <- fit$prob_distr_params[[1]][[2]]
  # 
  #   simulated_1 <- rmvnorm(n_sim, mean = mean_vec, sigma = sigma_mat)
  #   
  #   mean_scalar <- fit$prob_distr_params[[2]][[1]]
  #   var_scalar  <- fit$prob_distr_params[[2]][[2]]
  #   
  #   simulated_2 <- rnorm(n_sim, mean = mean_scalar, sd = sqrt(var_scalar))
  #   
  # } 
  # 
  # simulated_1 <- as.data.frame(simulated_1)
  # simulated_2 <- as.data.frame(simulated_2)
  # 
  # simulated = bind_cols(simulated_1, simulated_2)
  # 
  # m <- (-1 + sqrt(1 + 8*ncol(simulated_1)))/2
  # degmix_clustering_names <- c()
  # for (i in (seq_len(m))) {
  #   for (j in 1:(i)) {
  #     degmix_clustering_names <- c(degmix_clustering_names, paste0("DM", j, i))
  #   }
  # }
  # degmix_clustering_names = c(degmix_clustering_names , "triangles")
  # colnames(simulated) <- degmix_clustering_names 
  #   
  # fit$theoretical <- list(
  #   theory_stats = simulated,
  #   type = "degmix_clustering"
  # )
  # 
  # return(fit)
  
  simulated_list <- list()
  
  # Loop through each statistic independently
  for (i in seq_along(fit$prob_distr)) {
    dist_name <- fit$prob_distr[i]
    params    <- fit$prob_distr_params[[i]]
    settings  <- .get_distr_settings(dist_name)
    
    # Each sampler returns its own matrix (n_sim x num_stats_for_this_dist)
    # We pass population/max_val in case this specific stat needs it
    simulated_list[[i]] <- settings$sampler(
      p = params, 
      n = n_sim, 
      population = fit$population,
      max_val = choose(fit$population, 2)
    )
  }
  
  # Combine all independent blocks into one wide matrix
  simulated <- as.data.frame(do.call(cbind, simulated_list))
  
  # --- Naming Logic ---
  # Block 1: Degree Mixing (Everything except the last column)
  mix_cols <- ncol(simulated) - 1
  m <- (-1 + sqrt(1 + 8 * mix_cols)) / 2
  
  degmix_names <- c()
  for (row in seq_len(m)) {
    for (col in 1:row) {
      degmix_names <- c(degmix_names, paste0("DM", col, row))
    }
  }
  
  # Block 2: Clustering (The last column)
  colnames(simulated) <- c(degmix_names, "triangles")
  
  fit$theoretical <- list(
    theory_stats = simulated,
    type = "degmix_clustering"
  )
  
  return(fit)
}