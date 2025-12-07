#' CCM_theoretical_check: Generate theoretical statistics
#'
#' Adds theoretical distribution results to a CCM_fit object.
#'
#' @export
CCM_theoretical_check <- function(
    fit,
    n_sim = nrow(fit$mcmc_stats)
) {
  stat <- fit$Network_stats[[1]]
  
  #---------------------------
  # Case 1: Mixing
  #---------------------------
  if (stat == "Mixing" && fit$Prob_Distr[[1]] == "Multinomial_Poisson") {
    
    lambda <- fit$Prob_Distr_Params[[1]][1]
    probs  <- fit$Prob_Distr_Params[[2]]
    
    # Simulate Poisson-Multinomial draws
    simulated <- matrix(NA, nrow = n_sim, ncol = length(probs))
    
    for (i in seq_len(n_sim)) {
      total_edges <- rpois(1, lambda)
      simulated[i, ] <- rmultinom(1, size = total_edges, prob = probs)
    }
    
    simulated <- as.data.frame(simulated)
    colnames(simulated) <- c("M11", "M12", "M22")
    
    fit$theoretical <- list(
      theory_stats = simulated,
      type = "Mixing"
    )
    
    return(fit)
  }
  
  #---------------------------
  # Case 2: Edge count with NP distribution
  #---------------------------
  if (stat == "Edge" && fit$Prob_Distr[[1]] == "NP") {
    
    pmf <- fit$Prob_Distr_Params[[1]]
    max_edges <- choose(fit$population, 2)
    
    if (length(pmf) != max_edges + 1) {
      stop("Prob_Distr_Params does not match edge PMF length.")
    }
    
    edges <- sample(0:max_edges, size = n_sim, replace = TRUE, prob = pmf)
    df <- data.frame(edges = edges)
    
    fit$theoretical <- list(
      theory_stats = df,
      type = "Edge"
    )
    
    return(fit)
  }
  
  #---------------------------
  # Case 3: Degree — not implemented yet
  #---------------------------
  if (stat == "Degree") {
    if (Prob_Distr[[1]] != "Multinomial_Poisson") {
      warning("Theoretical degree distribution currently implemented only for Multinomial_Poisson. Returning NULL.")
      fit$theoretical <- list(
        theory_stats = NULL,
        type = "Degree"
      )
      return(fit)
    }
    
    # Generate theoretical degree distribution
    n_sim <- nrow(fit$mcmc_stats)
    population <- ncol(fit$mcmc_stats)
    
    # theoretical_pmf is Prob_Distr_Params[[2]], as in your previous code
    theoretical_pmf <- Prob_Distr_Params[[2]]
    
    # Generate multinomial-Poisson theoretical samples
    set.seed(123)  # for reproducibility
    multi_pois <- t(rmultinom(n_sim, population, prob = theoretical_pmf))
    
    # Convert to data.frame and store in fit$theoretical
    df <- as.data.frame(multi_pois)
    colnames(df) <- paste0("deg", 0:(population-1))
    
    fit$theoretical <- list(
      theory_stats = df,
      type = "Degree"
    )
    
    return(fit)
  }
  
  stop("Theoretical distribution not implemented for this statistic.")
}
