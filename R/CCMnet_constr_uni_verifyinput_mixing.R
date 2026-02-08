#' Verify input
#'
#' @noRd

CCMnet_constr_uni_verifyinput_mixing <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                                           population, covPattern, remove_var_last_entry) {
  
  error = 0
  
  k <- length(unique(covPattern))
  num_params <- (k * (k + 1)) / 2  
  total_pair_indices <- num_params * 2 
  
  block1 <- unlist(lapply(1:k, seq_len))
  block2 <- rep(1:k, times = 1:k)
  
  # 5. Build the inputs vector
  inputs <- c(
    0, 1, 0,                      # input[0:2]
    total_pair_indices,           # input[3]: e.g., 42 for k=6
    num_params,                   # input[4]: e.g., 21 for k=6
    total_pair_indices + population, # input[5]: offset (42 + 1461 = 1503)
    block1,                       # The "From" indices
    block2,                       # The "To" indices
    covPattern                    # The data (starts at index 6 + total_pair_indices)
  )
  
  #inputs <- c(c(0, 1, 0), c(6,3,6 + population), c(1,1,2,1,2,2), covPattern)
  
  eta0 = rep(-999.5,1 + num_params)
  
  
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