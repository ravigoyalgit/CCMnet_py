#' Verify input
#'
#' @noRd

CCMnet_constr_uni_verifyinput_degdist <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                  population, covPattern, 
                                                  mean_vector, var_vector, prob_type_sub_code,
                                                  mean_vector_size, var_vector_size) {
  

  prob_type = c(1, 0, 0, 0, 1,prob_type_sub_code, mean_vector_size, var_vector_size)
  
  CCM_constr_info <- list(
    error = 0,
    prob_type = prob_type,
    mean_vector = mean_vector,
    var_vector = var_vector,
    Clist_nterms = 2, 
    Clist_fnamestring = "edges degree",
    Clist_snamestring = "CCMnet CCMnet",
    inputs = c(c(0,1,0,0), length(mean_vector), length(mean_vector), c(0:(length(mean_vector)-1))),
    eta0 = rep(-999.5,length(c(1, mean_vector,0))),
    stats = NULL,
    MHproposal_name = "TNT",
    MHproposal_package = "CCMnet"
  )
  return(CCM_constr_info)
}
