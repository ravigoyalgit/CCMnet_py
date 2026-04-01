#' Verify input
#'
#' @noRd

CCMnet_constr_uni_verifyinput_edges <- function(network_stats, prob_distr, prob_distr_params,
                                    population, covPattern,
                                    mean_vector, var_vector, prob_type_sub_code,
                                    mean_vector_size, var_vector_size) {
  
  if (network_stats == "edges") {
    prob_type <- c(c(0,0,0,0,1),prob_type_sub_code, mean_vector_size, var_vector_size)
  } 
  
  if (network_stats == "density") {
    prob_type <- c(c(0,0,0,0,2),prob_type_sub_code, mean_vector_size, var_vector_size)
  }
  
  # 4. Return formatted list for C
  return(list(
    error = 0,
    prob_type = prob_type,
    mean_vector = mean_vector,
    var_vector = var_vector,
    Clist_nterms = 2, 
    Clist_fnamestring = "edges nfstab",
    Clist_snamestring = "CCMnet CCMnet",
    inputs = c(0,1,0,0,1,0),
    eta0 = c(-999.5, -999.5),
    stats = NULL,
    MHproposal_name = "TNT",
    MHproposal_package = "CCMnet"
  ))
}