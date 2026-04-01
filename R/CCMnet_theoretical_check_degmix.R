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
  
  settings <- .get_distr_settings(fit$prob_distr[[1]])
  
  # Generate the multivariate draws
  simulated_mat <- settings$sampler(
    p = fit$prob_distr_params[[1]], 
    n = n_sim
  )
  
  simulated <- as.data.frame(simulated_mat)
  
  # Calculate number of groups 'm' based on the triangular number formula
  # Columns = m * (m + 1) / 2
  m <- (-1 + sqrt(1 + 8 * ncol(simulated))) / 2
  
  # Apply naming logic: DM11, DM12, DM22, etc.
  degmix_names <- c()
  for (i in seq_len(m)) {
    for (j in 1:i) {
      degmix_names <- c(degmix_names, paste0("DM", j, i))
    }
  }
  
  colnames(simulated) <- degmix_names 
  
  fit$target_distr <- list(
    target_stats = simulated,
    type = "degmix"
  )
  
  return(fit)
}