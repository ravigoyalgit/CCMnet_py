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
  
  fit$target_distr <- list(
    target_stats = simulated,
    type = "degmixing_triangles"
  )
  
  return(fit)
}