#' Verify input
#'
#' @noRd

CCMnet_constr_uni_verifyinput_mixing_degdist <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                   population, covPattern, remove_var_last_entry) {
  
  error = 0
  if (Network_stats[1] == "Mixing") { #swap prob_distr_params
    Prob_Distr_Params_temp = Prob_Distr_Params[[1]]
    Prob_Distr_Params[[1]] = Prob_Distr_Params[[2]]
    Prob_Distr_Params[[2]] = Prob_Distr_Params_temp
  }
  if (length(Prob_Distr_Params[[1]][[1]][[1]]) != length(Prob_Distr_Params[[1]][[1]][[2]])) {
    stop("Current limitation requires mean degree distributions to be of equal length.")
    error = 1
  }
  # if (dim(Prob_Distr_Params[[1]][[2]][[1]])[1] != dim(Prob_Distr_Params[[1]][[2]][[2]])[1]) {
  #   print("Error: Current limitation requires covariance matrices to be of equal dimensions.")
  #   error = 1
  # }
  # if (dim(Prob_Distr_Params[[1]][[2]][[1]])[1] != dim(Prob_Distr_Params[[1]][[2]][[1]])[2]) {
  #   print("Error: Covariance matrix is not square.")
  #   error = 1
  # }
  # if (dim(Prob_Distr_Params[[1]][[2]][[2]])[1] != dim(Prob_Distr_Params[[1]][[2]][[2]])[2]) {
  #   print("Error: Covariance matrix is not square.")
  #   error = 1
  # }
  
  if ((Prob_Distr[1] == "Poisson") && ((Prob_Distr[2] == "Poisson"))) {
    covariate_list = covPattern
    
    inputs1 = c(rbind(c(0:(length(Prob_Distr_Params[[1]][[1]][[1]])-1)), rep(1,length(Prob_Distr_Params[[1]][[1]][[1]]))))
    inputs2 = c(rbind(c(0:(length(Prob_Distr_Params[[1]][[1]][[2]])-1)), rep(2,length(Prob_Distr_Params[[1]][[1]][[2]]))))
    
    inputs = c(c(0,1,0,0), length(Prob_Distr_Params[[1]][[1]][[1]]) + length(Prob_Distr_Params[[1]][[1]][[2]]),
               2*(length(Prob_Distr_Params[[1]][[1]][[1]])+length(Prob_Distr_Params[[1]][[1]][[2]])) + population, inputs1, inputs2, covariate_list, c(6,3,6 + population), c(1,2,2,2), covariate_list)
    eta0 = rep(-999.5,length(c(1, Prob_Distr_Params[[1]][[1]][[1]],Prob_Distr_Params[[1]][[1]][[2]],1,1)))
    
    mean_vector = c(Prob_Distr_Params[[1]][[1]][[1]], Prob_Distr_Params[[1]][[1]][[2]],  Prob_Distr_Params[[2]][[1]])
    
    var_vector = c(c(0,0),c(0,0), 0,0)
    
    prob_type = c(1,1,0,0,1)
    
  } else if ((Prob_Distr[1] == "Normal") && ((Prob_Distr[2] == "Normal"))) {
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
    
    inputs1 = c(rbind(c(0:(length(Prob_Distr_Params[[1]][[1]][[1]])-1)), rep(1,length(Prob_Distr_Params[[1]][[1]][[1]]))))
    inputs2 = c(rbind(c(0:(length(Prob_Distr_Params[[1]][[1]][[2]])-1)), rep(2,length(Prob_Distr_Params[[1]][[1]][[2]]))))
    
    inputs = c(c(0,1,0,0), length(Prob_Distr_Params[[1]][[1]][[1]]) + length(Prob_Distr_Params[[1]][[1]][[2]]),
               2*(length(Prob_Distr_Params[[1]][[1]][[1]])+length(Prob_Distr_Params[[1]][[1]][[2]])) + population, inputs1, inputs2, covariate_list, c(6,3,6 + population), c(1, 1, 2, 1, 2, 2), covariate_list)
    eta0 = rep(-999.5,length(c(1, Prob_Distr_Params[[1]][[1]][[1]],Prob_Distr_Params[[1]][[1]][[2]],1,1,1)))
    
  
    mean_vector = c(Prob_Distr_Params[[1]][[1]][[1]], Prob_Distr_Params[[1]][[1]][[2]],  Prob_Distr_Params[[2]][[1]])
    
    if (remove_var_last_entry == TRUE) {
      inverse_var_x1 = solve(Prob_Distr_Params[[1]][[2]][[1]][-length(Prob_Distr_Params[[1]][[1]][[1]]),-length(Prob_Distr_Params[[1]][[1]][[1]])])
      inverse_var_x1 = rbind(inverse_var_x1,0)
      inverse_var_x1 = cbind(inverse_var_x1,0)
      
      inverse_var_x2 = solve(Prob_Distr_Params[[1]][[2]][[2]][-length(Prob_Distr_Params[[1]][[1]][[2]]),-length(Prob_Distr_Params[[1]][[1]][[2]])])
      inverse_var_x2 = rbind(inverse_var_x2,0)
      inverse_var_x2 = cbind(inverse_var_x2,0)
    } else {
      inverse_var_x1 = solve(Prob_Distr_Params[[1]][[2]][[1]])
      inverse_var_x2 = solve(Prob_Distr_Params[[1]][[2]][[2]])
    }
    
    var_vector = c(c(inverse_var_x1),c(inverse_var_x2), Prob_Distr_Params[[2]][[2]])
    
    prob_type = c(1,1,0,0,1)
    
  } else if ((Prob_Distr[1] == "Tdist") && ((Prob_Distr[2] == "Tdist"))) {
    if (length(Prob_Distr_Params[[1]]) != 3) {
      
    }
    if (dim(Prob_Distr_Params[[1]][[3]][1]) > 0) {
      stop("Degrees of freedom are not greater than 0.")
      error = 1
    }
    if (dim(Prob_Distr_Params[[1]][[3]][2]) > 0) {
      stop("Degrees of freedom are not greater than 0.")
      error = 1
    }
    covariate_list = covPattern
    
    inputs1 = c(rbind(c(0:(length(Prob_Distr_Params[[1]][[1]][[1]])-1)), rep(1,length(Prob_Distr_Params[[1]][[1]][[1]]))))
    inputs2 = c(rbind(c(0:(length(Prob_Distr_Params[[1]][[1]][[2]])-1)), rep(2,length(Prob_Distr_Params[[1]][[1]][[2]]))))
    
    inputs = c(c(0,1,0,0), length(Prob_Distr_Params[[1]][[1]][[1]]) + length(Prob_Distr_Params[[1]][[1]][[2]]),
               2*(length(Prob_Distr_Params[[1]][[1]][[1]])+length(Prob_Distr_Params[[1]][[1]][[2]])) + population, inputs1, inputs2, covariate_list, c(4,2,4 + population), c(1,2,2,2), covariate_list)
    eta0 = rep(-999.5,length(c(1, Prob_Distr_Params[[1]][[1]][[1]],Prob_Distr_Params[[1]][[1]][[2]],1,1)))
    
    mean_vector = c(Prob_Distr_Params[[1]][[1]][[1]], Prob_Distr_Params[[1]][[1]][[2]],  Prob_Distr_Params[[2]][[1]], Prob_Distr_Params[[1]][[3]][1], Prob_Distr_Params[[1]][[3]][2], Prob_Distr_Params[[2]][[3]])
    
    if (remove_var_last_entry == TRUE) {
      inverse_var_x1 = solve(Prob_Distr_Params[[1]][[2]][[1]][-length(Prob_Distr_Params[[1]][[1]][[1]]),-length(Prob_Distr_Params[[1]][[1]][[1]])])
      inverse_var_x1 = rbind(inverse_var_x1,0)
      inverse_var_x1 = cbind(inverse_var_x1,0)
      
      inverse_var_x2 = solve(Prob_Distr_Params[[1]][[2]][[2]][-length(Prob_Distr_Params[[1]][[1]][[2]]),-length(Prob_Distr_Params[[1]][[1]][[2]])])
      inverse_var_x2 = rbind(inverse_var_x2,0)
      inverse_var_x2 = cbind(inverse_var_x2,0)
    } else {
      inverse_var_x1 = solve(Prob_Distr_Params[[1]][[2]][[1]])
      inverse_var_x2 = solve(Prob_Distr_Params[[1]][[2]][[2]])
    }
    
    var_vector = c(c(inverse_var_x1),c(inverse_var_x2), Prob_Distr_Params[[2]][[2]])
    
    prob_type = c(2,2,0,0,1)
    
  } else {
    stop("No such distribution for DEGREE DISTRIBUTION + MIXING currently implemented.")
    error = 1
  }
  
  if (error == 1) {
    CCM_constr_info <- list(
      error = 1,
      prob_type = NULL,
      mean_vector = NULL,
      var_vector = NULL,
      Clist_nterms = NULL,
      Clist_fnamestring = NULL,
      Clist_snamestring = NULL,
      inputs =  NULL,
      eta0 = NULL,
      stats = NULL,
      MHproposal_name = NULL,
      MHproposal_package = NULL
    )
  }
  if (error == 0) {
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
}
