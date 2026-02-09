#' Verify input
#'
#' @noRd

CCMnet_constr_uni_verifyinput_degdist <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                                            population, covPattern, remove_var_last_entry) {
  
  error = 0
  if (length(Prob_Distr_Params[[1]][[1]]) < 2) {
    stop("length of mean vector is less than 2")
    error = 1
  }
  if (Prob_Distr == "Normal") {
    mean_vector = Prob_Distr_Params[[1]][[1]]
    var_vector = Prob_Distr_Params[[1]][[2]]
    
    mean_vector = mean_vector / population
    var_vector = var_vector / population^2
    prob_type = c(1,0,0,0,1)
    if (dim(var_vector)[1] != dim(var_vector)[2]) {
      stop("Covariance matrix is not square")
      error = 1
    }
    if (dim(var_vector)[1] != length(mean_vector)) {
      stop("Dimension mismatch between covariance matrix and mean vector")
      error = 1
    }
    
    if (remove_var_last_entry == TRUE) {
      inverse_var_x = solve(var_vector[-length(mean_vector),-length(mean_vector)])
      inverse_var_x = rbind(inverse_var_x,0)
      inverse_var_x = cbind(inverse_var_x,0)
      var_vector = c(inverse_var_x)
    } else {
      var_vector = solve(var_vector)
    }
    
  } else if (Prob_Distr == "NegBin") {
    mean_vector = Prob_Distr_Params[[1]][[1]]
    var_vector = c(0,0)
    prob_type = c(2,0,0,0,1)
  } else if (Prob_Distr == "DirMult") {
    mean_vector = Prob_Distr_Params[[1]][[1]]
    var_vector = c(0,0)
    prob_type = c(3,0,0,0,1)
  } else {
    stop("No such distribution for DEGREE DISTRIBUTION currently implemented.")
    error = 1
  }
  if (error == 0) {

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
      Clist_fnamestring = "edges degree",
      Clist_snamestring = "CCMnet CCMnet",
      inputs = c(c(0,1,0,0), length(mean_vector), length(mean_vector), c(0:(length(mean_vector)-1))),
      eta0 = rep(-999.5,length(c(1, mean_vector,0))),
      stats = NULL,
      MHproposal_name = "TNT",
      MHproposal_package = "CCMnet"
    )
  }
}
