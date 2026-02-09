#' Verify input
#'
#' @noRd

CCMnet_constr_uni_verifyinput_degmixing <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                                              population, covPattern, remove_var_last_entry) {
  
  error = 0
  if (Prob_Distr == "Normal") {
    
    if (!inherits(Prob_Distr_Params[[1]][[1]], "numeric")) {
      stop("Mean degree mixing should be a vector representing upper triangle of degree mixing matrix.")
      error = 1
    }
    if (dim(Prob_Distr_Params[[1]][[2]])[1] != dim(Prob_Distr_Params[[1]][[2]])[2]) {
      stop("Covariance matrix is not square.")
      error = 1
    }
    if (length(Prob_Distr_Params[[1]][[1]]) != dim(Prob_Distr_Params[[1]][[2]])[2]) {
      stop("mean vector and covariance matrix are not similar dimensions.")
      error = 1
    }
    
    max_degree = floor(sqrt(2*length(upper.tri(Prob_Distr_Params[[1]][[1]], diag = TRUE))))
    
    m1 = matrix(c(1:max_degree), nrow = max_degree, ncol = max_degree)
    m1 = m1[upper.tri(m1, diag = TRUE)]
    
    m2 = t(matrix(c(1:max_degree), nrow = max_degree, ncol = max_degree))
    m2 = m2[upper.tri(m2, diag = TRUE)]
    
    inputs = c(c(0,1,0), c(((max_degree+1)*max_degree), ((max_degree+1)*max_degree*.5), (((max_degree+1)*max_degree)+1)))
    inputs = c(inputs, m1, m2, max_degree)
    
    eta0 = rep(-999.5,1 + .5*((max_degree+1)*max_degree))
    
    prob_type = c(0,0,1,0,1)
    
    mean_vector = Prob_Distr_Params[[1]][[1]]
    
    if (remove_var_last_entry == TRUE) {
      inverse_var_x = solve(Prob_Distr_Params[[1]][[2]] [-length(mean_vector),-length(mean_vector)])
      inverse_var_x = rbind(inverse_var_x,0)
      inverse_var_x = cbind(inverse_var_x,0)
    } else {
      inverse_var_x = solve(Prob_Distr_Params[[1]][[2]])
    }
    
    var_vector = c(inverse_var_x)
  } else {
    stop("No such distribution for DEGREE MIXING currently implemented.")
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
      Clist_nterms = 2, #Number of different terms
      Clist_fnamestring = "edges degmix",
      Clist_snamestring = "CCMnet CCMnet",
      inputs =inputs,
      eta0 = eta0,
      stats = NULL,
      MHproposal_name = "TNT",
      MHproposal_package = "CCMnet"
    )
  }
}
