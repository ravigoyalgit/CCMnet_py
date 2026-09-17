#' Verify input
#'
#' @noRd

CCMnet_constr_uni_verifyinput_degmixing_clustering <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                         population, covPattern, 
                                                         mean_vector, var_vector, prob_type_sub_code,
                                                         mean_vector_size, var_vector_size) {
  
    
    max_degree = floor(sqrt(2*length(upper.tri(Prob_Distr_Params[[1]][[1]], diag = TRUE))))
    
    m1 = matrix(c(1:max_degree), nrow = max_degree, ncol = max_degree)
    m1 = m1[upper.tri(m1, diag = TRUE)]
    
    m2 = t(matrix(c(1:max_degree), nrow = max_degree, ncol = max_degree))
    m2 = m2[upper.tri(m2, diag = TRUE)]
    
    inputs = c(c(0,1,0), c(((max_degree+1)*max_degree), ((max_degree+1)*max_degree*.5), (((max_degree+1)*max_degree)+1)))
    inputs = c(inputs, m1, m2, max_degree, c(0,1,0))
    
    eta0 = rep(-999.5, 1 + .5*((max_degree+1)*max_degree) + 1)

    prob_type = c(0,0,1,1,1,prob_type_sub_code[1],mean_vector_size[1], var_vector_size[1], prob_type_sub_code[2], mean_vector_size[2], var_vector_size[2])
    
    CCM_constr_info <- list(
      error = 0,
      prob_type = prob_type,
      mean_vector = mean_vector,
      var_vector = var_vector,
      Clist_nterms = 3, #Number of different terms
      Clist_fnamestring = "edges degmix triangle",
      Clist_snamestring = "CCMnet CCMnet CCMnet",
      inputs =inputs,
      eta0 = eta0,
      stats = NULL,
      MHproposal_name = "TNT",
      MHproposal_package = "CCMnet"
    )

}
