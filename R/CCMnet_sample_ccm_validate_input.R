#' Validate sample_ccm Inputs
#'
#' @noRd

.validate_sample_ccm_inputs <- function(network_stats, prob_distr, prob_distr_params, 
                                        population, sample_size, burnin, interval, 
                                        cov_pattern, initial_g, use_initial_g) {
  
  
  
  # 1. Numeric Scalars
  if (length(sample_size) != 1 || !is.numeric(sample_size) || sample_size %% 1 != 0 || sample_size < 2) {
    stop("`sample_size` must be a single integer >= 2")
  }
  if (length(burnin) != 1 || !is.numeric(burnin) || burnin %% 1 != 0 || burnin < 1) {
    stop("`burnin` must be a single integer >= 1")
  }
  if (length(interval) != 1 || !is.numeric(interval) || interval %% 1 != 0 || interval < 1) {
    stop("`interval` must be a single integer >= 1")
  }
  if (length(population) != 1 || !is.numeric(population) || population %% 1 != 0 || population < 2) {
    stop("`population` must be a single integer >= 2")
  }
  
  # 2. Covariate Pattern
  if (!is.null(cov_pattern)) {
    if (!is.numeric(cov_pattern) && !is.integer(cov_pattern)) stop("cov_pattern must be numeric.")
    if (length(cov_pattern) != population) {
      stop(sprintf("cov_pattern length (%d) must match population (%d).", 
                   length(cov_pattern), population))
    }
    if (any(is.na(cov_pattern))) stop("cov_pattern cannot contain NAs.")
  }
  
  # 3. Initial Graph
  if (use_initial_g) {
    if (is.null(initial_g) || !inherits(initial_g, "igraph")) {
      stop("When use_initial_g is TRUE, initial_g must be a valid igraph object.")
    }
    if (igraph::vcount(initial_g) != population) {
      stop("initial_g vertex count must match population.")
    }
  }
  
  # 4. network_stats
  if (any(!(network_stats %in% .VAL_NETPROPS))) {
    # Identify the specific invalid stats to help the user
    invalid_stats <- network_stats[!(network_stats %in% .VAL_NETPROPS)]
    
    stop(paste0(
      "invalid 'network_stats': ", paste(invalid_stats, collapse = ", "), 
      ". Must be one or more of: ", paste(.VAL_NETPROPS, collapse = ", ")
    ))
  }
  
  network_stats_comb = paste(network_stats, collapse = "_")
  if (!(network_stats_comb %in% .VAL_NETPROPS_COMB)) {

    stop(paste0(
      "invalid combination of 'network_stats': ", network_stats_comb, 
      ". Must be one of when combined: ", paste(.VAL_NETPROPS_COMB, collapse = ", ")
    ))
  }
  
  # 5. Check Prob Distr Params (Basic)
  if (!is.list(prob_distr_params)) {
    stop("prob_distr_params must be a list.")
  }
  
  if (length(network_stats) != length(prob_distr)) {
    stop(paste0("Mismatched input: 'network_stats' has length ", length(network_stats), 
                ", but 'prob_distr' has length ", length(prob_distr), "."))
  }
  
  if (length(network_stats) != length(prob_distr_params)) {
    stop(paste0("Mismatched input: 'network_stats' has length ", length(network_stats), 
                ", but 'prob_distr_params' has length ", length(prob_distr_params), "."))
  }
  
  # 6. Check Prob Distr Params (Details)
  for (i in seq_along(prob_distr)) {
    dist_name <- prob_distr[i]
    params    <- prob_distr_params[[i]]
    network_prop <- network_stats[i]
    
    # Fetch settings (including our new rules)
    settings <- .get_distr_settings(dist_name)
    
    if (!(network_prop %in% settings$valid_network_prop)) {
      stop(paste0(
        "invalid probabiity distribution ", dist_name, " for ", network_prop))
    }
    
    # Iterate through the rules for p1, p2, etc.
    for (p_key in names(settings$rules)) {
      p_idx <- as.numeric(gsub("p", "", p_key))
      val   <- params[[p_idx]]
      rules <- settings$rules[[p_key]]
      
      for (rule in rules) {
        # Handle comparison rules that need the first parameter as a target
        if (rule %in% c("match_length", "match_matrix_dim")) {
          .VAL_RULES[[rule]](val, paste(dist_name, p_key), params[[1]])
        } else {
          .VAL_RULES[[rule]](val, paste(dist_name, p_key))
        }
      }
    }
  }
  
  return(invisible(TRUE))
}
