#' Verify input
#'
#' @noRd

CCMnet_constr_uni_verifyinput_mixing_degdist <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                   population, covPattern, 
                                                   mean_vector, var_vector, prob_type_sub_code,
                                                   mean_vector_size, var_vector_size) {
  
    covariate_list = covPattern
    
    # 1. Degree Metadata (16 values)
    inputs_degree_meta = c(rbind(0:3, rep(1, 4)), rbind(0:3, rep(2, 4)))
    
    # 2. Mixing Metadata (6 values)
    inputs_mixing_meta = c(1, 1, 2, 1, 2, 2)
    
    # 3. Build the vector
    inputs = c(
      # Model Header
      c(0, 1, 0, 0), 
      
      # Term 1: Degree (8 stats, 116 total params)
      c(8, 116), 
      inputs_degree_meta, 
      covariate_list, # 100 attributes
      
      # Term 2: Nodemix (3 stats, 106 total params)
      c(6, 3, 106), 
      inputs_mixing_meta, 
      covariate_list # 100 attributes
    )
    
    inputs1 = c(rbind(c(0:(length(Prob_Distr_Params[[1]][[1]])-1)), rep(1,length(Prob_Distr_Params[[1]][[1]]))))
    inputs2 = c(rbind(c(0:(length(Prob_Distr_Params[[2]][[1]])-1)), rep(2,length(Prob_Distr_Params[[2]][[1]]))))
    
    inputs = c(c(0,1,0,0), length(Prob_Distr_Params[[1]][[1]]) + length(Prob_Distr_Params[[2]][[1]]),
               2*(length(Prob_Distr_Params[[1]][[1]])+length(Prob_Distr_Params[[2]][[1]])) + population, inputs1, inputs2, covariate_list, c(6,3,6 + population), c(1, 1, 2, 1, 2, 2), covariate_list)
    eta0 = rep(-999.5,length(c(1, Prob_Distr_Params[[1]][[1]],Prob_Distr_Params[[2]][[1]],1,1,1)))

  
  prob_type = c(1,1,0,0,1,prob_type_sub_code[1],mean_vector_size[1], var_vector_size[1], 
                prob_type_sub_code[2], mean_vector_size[2], var_vector_size[2], 
                prob_type_sub_code[3], mean_vector_size[3], var_vector_size[3])
  
    CCM_constr_info <- list(
      error = 0,
      prob_type = prob_type,
      mean_vector = mean_vector,
      var_vector = var_vector,
      Clist_nterms = 3, #Number of different terms
      Clist_fnamestring = "edges degree_by_attr nodemix",
      Clist_snamestring = "CCMnet CCMnet CCMnet",
      inputs = inputs,
      eta0 = eta0,
      stats = NULL,
      MHproposal_name = "TNT",
      MHproposal_package = "CCMnet"
    )
}
