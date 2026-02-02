#' Verify input
#'
#' @keywords internal

CCMnet_constr_uni_verifyinput_mixing <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                                           population, covPattern, remove_var_last_entry) {
  
  error = 0
  
  covariate_list = covPattern
  
  term1_header   <- c(0, 1, 0) # Edges: Offset 0, Stats 1, Params 0
  nodemix_params <- c(2, 0, 3, 0, 1, 2) # n_lev, directed, n_stats, base, lev1, lev2
  
  term2_header   <- c(
    length(nodemix_params),     # Offset to reach attributes
    3,                          # Number of stats
    length(nodemix_params) + population  # Total jump to reach the end of the model
  )
  
  #inputs <- c(term1_header, term2_header, nodemix_params, covariate_list)
  inputs <- c(c(0, 1, 0), c(6,3,6 + population), c(1,1,2,1,2,2), covariate_list)
  
  eta0 = rep(-999.5,length(c(1,1,1,1)))
  
  
  if (Prob_Distr[[1]] == 'Poisson') {
    mean_vector = Prob_Distr_Params[[1]][[1]]
    var_vector = c(0,0)
  
    prob_type = c(0,1,0,0,1)
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
      Clist_nterms = 2,
      Clist_fnamestring = "edges nodemix",
      Clist_snamestring = "CCMnet CCMnet",
      inputs = inputs,
      eta0 = eta0,
      stats = NULL,
      MHproposal_name = "TNT",
      MHproposal_package = "CCMnet"
    )
  }
  
}