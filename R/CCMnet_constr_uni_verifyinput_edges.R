#' Verify input
#'
#' @noRd

CCMnet_constr_uni_verifyinput_edges <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                                    population, covPattern, remove_var_last_entry) {
  
  error = 0
  if (Network_stats == "edges") {
    if (Prob_Distr == "normal") {
      if (length(Prob_Distr_Params[[1]][[1]]) != 1) {
        stop("mean for EDGES is not one value")
        error = 1
      } else if (length(Prob_Distr_Params[[1]][[2]]) != 1) {
        stop("variance for EDGES is not one value")
        error = 1
      } else if (Prob_Distr_Params[[1]][[1]] <= 0) {
        stop("mean for EDGES not a positive value")
        error = 1
      } else if (Prob_Distr_Params[[1]][[2]] <= 0) {
        stop("variance for EDGES not a positive value")
        error = 1
      }
      if (error == 0) {
        prob_type = c(0,0,0,0,1)
        mean_vector = c(Prob_Distr_Params[[1]][[1]],Prob_Distr_Params[[1]][[1]])
        var_vector = c(Prob_Distr_Params[[1]][[2]], Prob_Distr_Params[[1]][[2]])
      }
    } else if (Prob_Distr == "lognormal") {
      if (length(Prob_Distr_Params[[1]][[1]]) != 1) {
        stop("lambda for EDGES is not one value")
        error = 1
      } else if (Prob_Distr_Params[[1]][[1]] <= 0) {
        stop("lambda for EDGES not a positive value")
        error = 1
      } 
      if (error == 0) {
        prob_type = c(0,0,0,0,2)
        mean_vector = c(Prob_Distr_Params[[1]][[1]],Prob_Distr_Params[[1]][[1]])
        var_vector = c(0,0)
      }
    } else if (Prob_Distr == "poisson") {
      if (length(Prob_Distr_Params[[1]][[1]]) != 1) {
        stop("lambda for EDGES is not one value")
        error = 1
      } else if (Prob_Distr_Params[[1]][[1]] <= 0) {
        stop("lambda for EDGES not a positive value")
        error = 1
      } 
      if (error == 0) {
        prob_type = c(0,0,0,0,3)
        mean_vector = c(Prob_Distr_Params[[1]][[1]],Prob_Distr_Params[[1]][[1]])
        var_vector = c(0,0)
      }
    } else if (Prob_Distr == "uniform") {
      prob_type = c(0,0,0,0,4)
      mean_vector = c(1, 1)
      var_vector = c(0,0)
    } else if (Prob_Distr == "np") {
      if (length(Prob_Distr_Params[[1]][[1]]) != (choose(population,2)+1)) {
        stop("need probability for all possible values for EDGES")
        error = 1
      } else if (sum(Prob_Distr_Params[[1]][[1]]) != 1) {
        stop("probabilities much sum to 1")
        error = 1
      } 
      if (error == 0) {
        prob_type = c(0,0,0,0,99)
        mean_vector = Prob_Distr_Params[[1]][[1]]
        var_vector = c(0,0)
      }
    } else {
      stop("No such distribution for EDGES currently implemented.")
      error = 1
    } 
  }
  
  
  if (Network_stats == "Density") {
    if (Prob_Distr == "Normal") {
      prob_type = c(0,0,0,0,11)
      mean_vector = c(Prob_Distr_Params[[1]][[1]],Prob_Distr_Params[[1]][[1]])
      var_vector = c(Prob_Distr_Params[[1]][[2]], Prob_Distr_Params[[1]][[2]])
      if (length(Prob_Distr_Params[[1]][[1]]) != 1) {
        stop("mean value for network density is one positive value")
        error = 1
      }
      if (length(Prob_Distr_Params[[1]][[2]]) != 1) {
        stop("variance for network density is one positive value")
        error = 1
      }
    } else if (Prob_Distr == "Beta") {
      prob_type = c(0,0,0,0,12)
      mean_vector = c(Prob_Distr_Params[[1]][[1]],Prob_Distr_Params[[1]][[2]])
      var_vector = c(0,0)
    } else {
      stop("No such distribution for DENSITY currently implemented.")
      error = 1
    } 
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
      Clist_fnamestring = "edges nfstab",
      Clist_snamestring = "CCMnet CCMnet",
      inputs = c(0,1,0,0,1,0),
      eta0 = c(-999.5, -999.5),
      stats = NULL,
      MHproposal_name = "TNT",
      MHproposal_package = "CCMnet"
    )
  }
  return(CCM_constr_info)
}