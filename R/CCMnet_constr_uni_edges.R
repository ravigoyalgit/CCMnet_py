CCMnet_constr_uni_edges <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                                          nedges, g, max_degree,
                                          population, covPattern, remove_var_last_entry) {
  
  error = 0
  if (Prob_Distr == "Normal") {
    prob_type = c(0,0,0,0,1)
    mean_vector = c(Prob_Distr_Params[[1]][[1]],Prob_Distr_Params[[1]][[1]])
    var_vector = c(Prob_Distr_Params[[1]][[2]], Prob_Distr_Params[[1]][[2]])
    if (length(Prob_Distr_Params[[1]][[1]]) != 1) {
      print("Error: mean value for network density is one positive value")
      error = 1
    }
    if (length(Prob_Distr_Params[[1]][[2]]) != 1) {
      print("Error: variance for network density is one positive value")
      error = 1
    }
  } else if (Prob_Distr == "LogNormal") {
    prob_type = c(0,0,0,0,2)
    mean_vector = c(Prob_Distr_Params[[1]][[1]],Prob_Distr_Params[[1]][[1]])
  } else if (Prob_Distr == "Poisson") {
    prob_type = c(0,0,0,0,3)
    mean_vector = c(Prob_Distr_Params[[1]][[1]],Prob_Distr_Params[[1]][[1]])
  } else if (Prob_Distr == "Uniform") {
    prob_type = c(0,0,0,0,4)
    mean_vector = c(1, 1)
  } else if (Prob_Distr == "NP") {
    prob_type = c(0,0,0,0,99)
    mean_vector = Prob_Distr_Params[[1]][[1]]
    var_vector = c(0,0)
  } else {
    print("Error: No such distribution for degree distribution currently implemented.")
    print("Email ravi.goyal@mail.harvard.edu to add feature.")
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
      Clist_fnamestring = "edges nfstab",
      Clist_snamestring = "CCMnet CCMnet",
      inputs = c(0,1,0,0,1,0),
      eta0 = c(-999.5, -999.5),
      stats = c(nedges[1],nedges[1]),
      MHproposal_name = "TNT",
      MHproposal_package = "CCMnet"
    )
  }
  return(CCM_constr_info)
}