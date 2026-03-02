#' Verify input
#'
#' @noRd

CCMnet_constr_uni_verifyinput <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                                          population, covPattern, remove_var_last_entry) {
  
  # 1. Basic Length & Type Validation
  if (length(Network_stats) != length(Prob_Distr)) {
    stop(paste0("Mismatched input: 'Network_stats' has length ", length(Network_stats), 
                ", but 'Prob_Distr' has length ", length(Prob_Distr), "."))
  }
  
  if (!is.numeric(population) || population <= 1) {
    stop("population must be a positive integer greater than 1.")
  }
  
  # 2. Key Standardization
  stat_order <- order(Network_stats)
  stat_key   <- paste(Network_stats[stat_order], collapse = "+")
  distr_key  <- paste(Prob_Distr[stat_order], collapse = "+")
  
  # 3. Master Lookup Check
  valid_distr_options <- .get_supported_distrs(stat_key)
  
  if (length(valid_distr_options) == 0) {
    supported_stats <- paste(names(eval(body(.get_supported_distrs)[[2]])), collapse = ", ")
    
    stop(paste0(
      "The combination of statistics '", stat_key, "' is not implemented.\n",
      "Currently supported network properties: ", supported_stats, "."
    ))
  }
  
  if (!(distr_key %in% valid_distr_options)) {
    stop(paste0(
      "The distribution '", distr_key, "' is not supported for '", stat_key, "'.\n",
      "Supported options: ", paste(valid_distr_options, collapse = ", "), "."
    ))
  }
  
  # 4. Routing to Sub-functions
  
  # Logic for 1-statistic modes
  if (stat_key == "edges" || stat_key == "density") {
    CCM_info <- CCMnet_constr_uni_verifyinput_edges(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                    population, covPattern, remove_var_last_entry)
    
  } else if (stat_key == "mixing") {
    CCM_info <- CCMnet_constr_uni_verifyinput_mixing(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                     population, covPattern, remove_var_last_entry)
    
  } else if (stat_key == "degreedist") {
    CCM_info <- CCMnet_constr_uni_verifyinput_degdist(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                      population, covPattern, remove_var_last_entry)
    
  } else if (stat_key == "degmixing") {
    CCM_info <- CCMnet_constr_uni_verifyinput_degmixing(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                        population, covPattern, remove_var_last_entry)
    
  # Logic for 2-statistic modes
  } else if (stat_key == "degreedist+mixing") {
    CCM_info <- CCMnet_constr_uni_verifyinput_mixing_degdist(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                             population, covPattern, remove_var_last_entry)
    
  } else if (stat_key == "degmixing+triangles") {
    CCM_info <- CCMnet_constr_uni_verifyinput_degmixing_clustering(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                                   population, covPattern, remove_var_last_entry)
    
  } 
  
  return(CCM_info)
}
