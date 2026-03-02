#' Verify input
#'
#' @noRd

CCMnet_constr_uni_verifyinput_edges <- function(network_stats, prob_distr, prob_distr_params,
                                    population, covPattern, remove_var_last_entry) {
  
  # 1. Validation for EDGES
  if (network_stats == "edges") {
    params <- prob_distr_params[[1]]
    
    if (prob_distr == "normal") {
      mu <- params[[1]]
      sigma2 <- params[[2]]
      
      if (length(mu) != 1 || !is.numeric(mu) || mu < 0) 
        stop("The 'mean' for EDGES must be a single non-negative numeric value.")
      if (length(sigma2) != 1 || !is.numeric(sigma2) || sigma2 <= 0) 
        stop("The 'variance' for EDGES must be a single positive numeric value.")
      
      prob_type <- c(0,0,0,0,1)
      mean_vector <- c(mu, mu)
      var_vector <- c(sigma2, sigma2)
      
    } else if (prob_distr == "poisson" || prob_distr == "lognormal") {
      lambda <- params[[1]]
      if (length(lambda) != 1 || !is.numeric(lambda) || lambda <= 0) 
        stop(paste("The parameter for", toupper(prob_distr), "EDGES must be a single positive numeric value."))
      
      type_code <- ifelse(prob_distr == "lognormal", 2, 3)
      prob_type <- c(0,0,0,0, type_code)
      mean_vector <- c(lambda, lambda)
      var_vector <- c(0, 0)
      
    } else if (prob_distr == "uniform") {
      prob_type <- c(0,0,0,0,4)
      mean_vector <- c(1, 1)
      var_vector <- c(0,0)
      
    } else if (prob_distr == "np") {
      probs <- params[[1]]
      max_edges <- choose(population, 2)
      if (length(probs) != (max_edges + 1)) 
        stop(sprintf("Non-parametric (np) EDGES requires a probability vector of length %d (0 to max edges).", max_edges + 1))
      if (abs(sum(probs) - 1) > 1e-8) 
        stop("Non-parametric probabilities for EDGES must sum to 1.")
      
      prob_type <- c(0,0,0,0,99)
      mean_vector <- probs
      var_vector <- c(0,0)
      
    }
  }
  
  # 2. Validation for DENSITY
  if (network_stats == "density") {
    params <- prob_distr_params[[1]]
    
    if (prob_distr == "normal") {
      mu <- params[[1]]
      sigma2 <- params[[2]]
      
      if (length(mu) != 1 || mu < 0 || mu > 1) 
        stop("The 'mean' for DENSITY must be a numeric value between 0 and 1.")
      if (length(sigma2) != 1 || sigma2 <= 0) 
        stop("The 'variance' for DENSITY must be a single positive numeric value.")
      
      prob_type <- c(0,0,0,0,11)
      mean_vector <- c(mu, mu)
      var_vector <- c(sigma2, sigma2)
      
    } else if (prob_distr == "beta") {
      shape1 <- params[[1]]
      shape2 <- params[[2]]
      
      if (length(shape1) != 1 || shape1 <= 0 || length(shape2) != 1 || shape2 <= 0)
        stop("BETA distribution for DENSITY requires positive alpha and beta shape parameters.")
      
      prob_type <- c(0,0,0,0,12)
      mean_vector <- c(shape1, shape2)
      var_vector <- c(0,0)
      
    }
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