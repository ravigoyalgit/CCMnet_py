CCM <- function(null_model,
                alt_model = NULL,
                population,
                Prob_Distr,
                Prob_Distr_Params,
                covPattern,
                samplesize,
                burnin,
                interval,
                test_stat = c("hellinger")) {
  
  ## ---- Parse formulas ----
  if (!inherits(null_model, "formula")) {
    stop("null_model must be a formula of the form g ~ edges")
  }
  
  lhs <- null_model[[2]]
  g_obs <- eval(lhs, envir = parent.frame())
  
  if (!inherits(g_obs, "igraph")) {
    stop("Left-hand side of null_model must be an igraph object")
  }
  
  null_terms <- attr(terms(null_model), "term.labels")
  alt_terms  <- if (!is.null(alt_model)) attr(terms(alt_model), "term.labels") else character(0)
  
  ## ---- Fit null CCM ----
  fit_null <- CCM_fit(
    Network_stats = list(stats_from_terms(null_terms)),
    Prob_Distr = Prob_Distr,
    Prob_Distr_Params = Prob_Distr_Params,
    population = population,
    covPattern = covPattern,
    samplesize = samplesize,
    burnin = burnin,
    interval = interval
  )
  
  ## ---- Extract observed statistics ----
  obs_stats <- extract_observed_stats(g_obs, population)
  
  ## ---- Compute test for each alt term ----
  tests <- lapply(alt_terms, function(term) {
    
    if (term == "degree") {
      
      degree_samples <- fit_null$mcmc_stats[, paste0("deg", 0:(population - 1))]
      degree_obs <- table(factor(degree(g_obs), levels = 0:(population - 1)))
      
      E_N <- colMeans(degree_samples)
      
      hellinger <- function(p, q) {
        sqrt(sum((sqrt(p) - sqrt(q))^2)) / sqrt(2)
      }
      
      T_obs <- hellinger(
        degree_obs / sum(degree_obs),
        E_N / sum(E_N)
      )
      
      T_null <- apply(degree_samples, 1, function(Ns) {
        hellinger(
          Ns / sum(Ns),
          E_N / sum(E_N)
        )
      })
      
      p_value <- mean(T_null >= T_obs)
      
      list(
        term = term,
        statistic = "hellinger",
        T_obs = T_obs,
        T_null = T_null,
        p_value = p_value
      )
    } else {
      stop(paste("Unsupported term:", term))
    }
  })
  
  names(tests) <- alt_terms
  
  alt_terms <- attr(terms(alt_model), "term.labels")
  
  ## ---- Return object ----
  structure(
    list(
      call = match.call(),
      null_model = null_model,
      alt_model = alt_model,
      alt_terms  = alt_terms,
      observed_network = g_obs,
      fit_null = fit_null,
      tests = tests
    ),
    class = "ccm_inference"
  )
}

stats_from_terms <- function(terms) {
  if ("edges" %in% terms) "Edge" else character(0)
}

extract_observed_stats <- function(g, population) {
  list(
    edges = gsize(g),
    degree = table(factor(degree(g), levels = 0:(population - 1)))
  )
}

print.ccm_inference <- function(x, ...) {
  cat("CCM inference for network data\n\n")
  cat("Null model:  ", deparse(x$null_model), "\n")
  if (!is.null(x$alt_model)) {
    cat("Tested terms:", names(x$tests), "\n\n")
  }
  for (t in x$tests) {
    cat("Term:", t$term, "\n")
    cat("Test statistic:", t$statistic, "\n")
    cat("p-value:", signif(t$p_value, 3), "\n\n")
  }
}

plot.ccm_inference <- function(x,
                               type = c("test", "network"),
                               term = NULL,
                               ...) {
  
  type <- match.arg(type)
  
  # Default term = first alternative term
  if (is.null(term)) {
    term <- x$alt_terms[1]
  }
  
  if (!term %in% names(x$tests)) {
    stop("No test available for term: ", term)
  }
  
  if (type == "test") {
    return(plot_ccm_test(x, term))
  }
  
  if (type == "network") {
    return(plot_ccm_network(x, term))
  }
}

plot_ccm_test <- function(x, term) {
  
  test <- x$tests[[term]]
  
  df <- data.frame(T = test$T_null)
  
  ggplot(df, aes(x = T)) +
    geom_density() +
    geom_vline(xintercept = test$T_obs, linetype = "dashed") +
    labs(
      title = paste("Null distribution of", term, "test statistic"),
      subtitle = paste("p-value =", signif(test$p_value, 3)),
      x = test$statistic
    ) +
    theme_minimal()
}

plot_ccm_network <- function(x, term) {
  
  if (term == "degree") {
    return(plot_ccm_degree(x))
  }
  
  stop("No network plot implemented for term: ", term)
}

plot_ccm_degree <- function(x) {
  
  population <- x$fit_null$population
  
  # Observed
  degree_obs <- degree(x$observed_network)
  degree_counts_obs <- table(factor(degree_obs, levels = 0:(population - 1)))
  
  # CCM mean
  degree_samples <- x$fit_null$mcmc_stats[, paste0("deg", 0:(population - 1))]
  degree_mean <- colMeans(degree_samples)
  
  df <- data.frame(
    degree = 0:(population - 1),
    observed = as.numeric(degree_counts_obs),
    posterior_mean = degree_mean
  )
  
  ggplot(df, aes(x = degree)) +
    geom_line(aes(y = observed, color = "Observed")) +
    geom_line(aes(y = posterior_mean, color = "CCM Posterior Mean")) +
    labs(
      y = "Number of nodes",
      color = "",
      title = "Degree distribution: observed vs CCM"
    ) +
    theme_minimal()
}


