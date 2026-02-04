#' Run CCM Hypothesis Tests
#'
#' @noRd
run_ccm_tests <- function(g_obs,
                          fit_null,
                          alt_terms,
                          population,
                          test_stat) {
  
  tests <- lapply(alt_terms, function(term) {
    
    if (term == "DegreeDist") {
      return(test_degree_ccm(
        g_obs, fit_null, population, test_stat
      ))
    }
    
    stop("Unsupported term: ", term)
  })
  
  names(tests) <- alt_terms
  tests
}

#' Degree-Based CCM Test
#'
#' @noRd
test_degree_ccm <- function(g_obs,
                            fit_null,
                            population,
                            test_stat) {
  
  degree_samples <- fit_null$mcmc_stats[, paste0("deg", 0:(population - 1))]
  degree_obs <- table(factor(igraph::degree(g_obs),
                             levels = 0:(population - 1)))
  
  E_N <- colMeans(degree_samples)
  
  T_obs <- hellinger_distance(
    degree_obs / sum(degree_obs),
    E_N / sum(E_N)
  )
  
  T_null <- apply(degree_samples, 1, function(Ns) {
    hellinger_distance(
      Ns / sum(Ns),
      E_N / sum(E_N)
    )
  })
  
  list(
    term = "degree",
    statistic = test_stat,
    T_obs = T_obs,
    T_null = T_null,
    p_value = mean(T_null >= T_obs)
  )
}
