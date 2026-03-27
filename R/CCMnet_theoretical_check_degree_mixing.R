
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
  
  # if (fit$prob_distr[[1]] == "mvn") {
  #   degrees_0.df <- rmvnorm(n = n_sim, 
  #           mean = fit$prob_distr_params[[1]][[1]],
  #           sigma = fit$prob_distr_params[[1]][[2]])
  #   degrees_1.df <- rmvnorm(n = n_sim, 
  #                           mean = fit$prob_distr_params[[2]][[1]],
  #                           sigma = fit$prob_distr_params[[2]][[2]])
  #   mixing.df = rnorm(n = n_sim, 
  #                     mean = fit$prob_distr_params[[3]][[1]],
  #                     sd = sqrt(fit$prob_distr_params[[3]][[2]]))
  #   
  #   degrees_mixing.df = bind_cols(bind_cols(degrees_0.df, degrees_1.df),
  #                                 mixing.df)
  #   
  #   df <- as.data.frame(degrees_mixing.df)
  #   len_deg = length(fit$prob_distr_params[[1]][[1]])
  #   cov0_names = paste(paste0("deg", 0:(len_deg-1)), "_1", sep = "")
  #   cov1_names = paste(paste0("deg", 0:(len_deg-1)), "_2", sep = "")
  #   mix_names = c("M21")
  #   colnames(df) <- c(cov0_names, cov1_names, mix_names)
  # 
  # } else {
  #   warning("Theoretical distribution not currently implemented. Returning NULL.")
  #   fit$theoretical <- list(
  #     theory_stats = NULL,
  #     type = "degree_mixing"
  #   )
  #   return(fit)
  # }
  # 
  # # Convert to data.frame and store in fit$theoretical
  # 
  # 
  # fit$theoretical <- list(
  #   theory_stats = df,
  #   type = "degree_mixing"
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
  df <- bind_cols(simulated_list)
  
  # --- Naming Logic ---
  # Block 1 & 2: Degree Distributions
  len_deg <- length(fit$prob_distr_params[[1]][[1]])
  cov0_names <- paste0("deg", 0:(len_deg - 1), "_1")
  cov1_names <- paste0("deg", 0:(len_deg - 1), "_2")
  
  # Block 3: Mixing Statistics (Remaining columns)
  mix_names <- "M21"
  
  colnames(df) <- c(cov0_names, cov1_names, mix_names)
  
  # 5. Store and Return
  fit$theoretical <- list(
    theory_stats = df,
    type = "degreedist_mixing"
  )
  
  return(fit)
}