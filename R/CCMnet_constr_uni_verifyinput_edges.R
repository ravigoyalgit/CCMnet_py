#' Verify input
#'
#' @noRd

CCMnet_constr_uni_verifyinput_edges <- function(network_stats, prob_distr, prob_distr_params,
                                    population, covPattern,
                                    mean_vector, var_vector, prob_type_sub_code,
                                    mean_vector_size, var_vector_size) {
  
  # # 1. Validation for EDGES
  # if (network_stats == "edges") {
  #   params <- prob_distr_params[[1]]
  #   
  #   if (prob_distr == "normal") {
  #     mu <- params[[1]]
  #     sigma2 <- params[[2]]
  #     
  #     prob_type <- c(0,0,0,0,1,1)
  #     mean_vector <- c(mu, mu)
  #     var_vector <- c(sigma2, sigma2)
  #     
  #   } else if (prob_distr == "poisson" || prob_distr == "lognormal") {
  #     lambda <- params[[1]]
  # 
  #     type_code <- ifelse(prob_distr == "lognormal", 2, 3)
  #     prob_type <- c(0,0,0,0, 1, type_code)
  #     mean_vector <- c(lambda, lambda)
  #     var_vector <- c(0, 0)
  #     
  #   } else if (prob_distr == "uniform") {
  #     prob_type <- c(0,0,0,0,1, 4)
  #     mean_vector <- c(1, 1)
  #     var_vector <- c(0,0)
  #     
  #   } else if (prob_distr == "np") {
  #     probs <- params[[1]]
  #     max_edges <- choose(population, 2)
  # 
  #     prob_type <- c(0,0,0,0,1,99)
  #     mean_vector <- probs
  #     var_vector <- c(0,0)
  #     
  #   }
  # }
  # 
  # # 2. Validation for DENSITY
  # if (network_stats == "density") {
  #   params <- prob_distr_params[[1]]
  #   
  #   if (prob_distr == "normal") {
  #     mu <- params[[1]]
  #     sigma2 <- params[[2]]
  #     
  #     prob_type <- c(0,0,0,0,2,1)
  #     mean_vector <- c(mu, mu)
  #     var_vector <- c(sigma2, sigma2)
  #     
  #   } else if (prob_distr == "beta") {
  #     shape1 <- params[[1]]
  #     shape2 <- params[[2]]
  # 
  #     prob_type <- c(0,0,0,0,2,5)
  #     mean_vector <- c(shape1, shape1)
  #     var_vector <- c(shape2, shape2)
  #     
  #   }
  # }
  
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