#' Verify input
#'
#' @noRd

CCMnet_constr_uni_verifyinput <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                                          population, covPattern, remove_var_last_entry) {
  
  # if (Network_stats[1] == "triangles") { #swap prob_distr_params
  #   Prob_Distr_Params_temp = Prob_Distr_Params[[1]]
  #   Prob_Distr_Params[[1]] = Prob_Distr_Params[[2]]
  #   Prob_Distr_Params[[2]] = Prob_Distr_Params_temp
  # }
  # 
  # if (Network_stats[1] == "mixing") { #swap prob_distr_params
  #   Prob_Distr_Params_temp = Prob_Distr_Params[[1]]
  #   Prob_Distr_Params[[1]] = Prob_Distr_Params[[2]]
  #   Prob_Distr_Params[[2]] = Prob_Distr_Params_temp
  # }

  stat_key = paste(Network_stats, collapse = "+")
  
  mean_vector <- NULL
  var_vector <- NULL
  prob_type_sub_code <- NULL
  mean_vector_size <- NULL
  var_vector_size <- NULL
  
  for (i in c(1:length(Network_stats))) {
    
    mean_vector_TEMP <- NULL
    var_vector_TEMP <- NULL
    
    # 1. Retrieve the config
    prob_distr_config <- .get_distr_settings(Prob_Distr[[i]])
    
    # 2. Extract and transform parameters
    if (prob_distr_config$mean_bool) {
      mean_vector_TEMP <- Prob_Distr_Params[[i]][[1]]
    } 
    
    if (prob_distr_config$var_bool) {
      var_vector_TEMP  <- Prob_Distr_Params[[i]][[2]]
    } 
    
    if (prob_distr_config$use_solve_var) {
      var_vector_TEMP <- c(solve(var_vector_TEMP))
    }
    mean_vector = c(mean_vector, mean_vector_TEMP)
    var_vector = c(var_vector, var_vector_TEMP)
    prob_type_sub_code <- c(prob_type_sub_code, prob_distr_config$sub_code)
    mean_vector_size <- c(mean_vector_size, length(mean_vector_TEMP))
    var_vector_size <- c(var_vector_size, length(var_vector_TEMP))
  }
  
  if (is.null(mean_vector)) {
    mean_vector = c(0,0)
  }
  
  if (is.null(var_vector)) {
    var_vector = c(0,0)
  }
  
  if (length(mean_vector) == 1) {
    mean_vector = c(mean_vector, mean_vector)
  }
  
  if (length(var_vector) == 1) {
    var_vector = c(var_vector, var_vector)
  }

  # Logic for 1-statistic modes
  if (stat_key == "edges" || stat_key == "density") {
    CCM_info <- CCMnet_constr_uni_verifyinput_edges(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                    population, covPattern, 
                                                    mean_vector, var_vector, prob_type_sub_code,
                                                    mean_vector_size, var_vector_size)
    
  } else if (stat_key == "mixing") {
    CCM_info <- CCMnet_constr_uni_verifyinput_mixing(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                     population, covPattern,
                                                     mean_vector, var_vector, prob_type_sub_code,
                                                     mean_vector_size, var_vector_size)
    
  } else if (stat_key == "degreedist") {
    CCM_info <- CCMnet_constr_uni_verifyinput_degdist(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                      population, covPattern, 
                                                      mean_vector, var_vector, prob_type_sub_code,
                                                      mean_vector_size, var_vector_size)
    
  } else if (stat_key == "degmixing") {
    CCM_info <- CCMnet_constr_uni_verifyinput_degmixing(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                        population, covPattern, 
                                                        mean_vector, var_vector, prob_type_sub_code,
                                                        mean_vector_size, var_vector_size)
    
  # Logic for 2-statistic modes
  } else if (stat_key == "degreedist+degreedist+mixing") {
    CCM_info <- CCMnet_constr_uni_verifyinput_mixing_degdist(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                             population, covPattern, 
                                                             mean_vector, var_vector, prob_type_sub_code,
                                                             mean_vector_size, var_vector_size)
    
  } else if (stat_key == "degmixing+triangles") {
    CCM_info <- CCMnet_constr_uni_verifyinput_degmixing_clustering(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                                   population, covPattern, 
                                                                   mean_vector, var_vector, prob_type_sub_code,
                                                                   mean_vector_size, var_vector_size)
    
  } 
  
  return(CCM_info)
}
