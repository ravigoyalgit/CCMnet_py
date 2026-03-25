#' Verify input
#'
#' @noRd

CCMnet_constr_uni_verifyinput_degdist <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                  population, covPattern, remove_var_last_entry) {
  
  params <- Prob_Distr_Params[[1]]
  mean_vector_raw <- params[[1]]
  
  # 1. Basic length check for degree distribution vector
  if (length(mean_vector_raw) < 2) {
    stop("Length of the mean vector for DEGREE DISTRIBUTION must be at least 2.")
  }
  
  if (Prob_Distr == "mvn") {
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
    
  } else if (Prob_Distr == "negbin") {
    mean_vector <- mean_vector_raw
    var_vector  <- c(0, 0)
    prob_type   <- c(2, 0, 0, 0, 1)
    
  } else if (Prob_Distr == "dirmult") {
    
    # Check for non-negativity and strictly positive values
    # Dirichlet alphas must be > 0 for the density to be well-defined
    if (any(mean_vector_raw <= 0)) {
      stop("All parameters for 'dirmult' DEGREE DISTRIBUTION must be strictly positive (> 0).")
    }
    
    # Check for infinite or NA values
    if (any(!is.finite(mean_vector_raw))) {
      stop("Parameters for 'dirmult' DEGREE DISTRIBUTION must be finite (no NA or Inf).")
    }
    
    mean_vector <- mean_vector_raw
    var_vector  <- c(0, 0)
    prob_type   <- c(1, 0, 0, 0, 1, 6)
  }
  
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
  return(CCM_constr_info)
}
