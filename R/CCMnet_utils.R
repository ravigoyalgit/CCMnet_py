#' Fit Null CCM
#'
#' @noRd
fit_ccm_null <- function(null_terms,
                         population,
                         Prob_Distr,
                         Prob_Distr_Params,
                         covPattern,
                         samplesize,
                         burnin,
                         interval,
                         alt_terms) {
  
  sample_ccm(
    network_stats = list(null_terms),
    prob_distr = Prob_Distr,
    prob_distr_params = Prob_Distr_Params,
    population = population,
    cov_pattern = covPattern,
    sample_size = samplesize,
    burnin = burnin,
    interval = interval,
    Obs_stats = list(alt_terms)
  )
}

#' Map Formula Terms to CCM Statistics
#'
#' @noRd
stats_from_terms <- function(terms) {
  if ("edges" %in% terms) "Edge" else character(0)
}

#' Hellinger Distance
#'
#' @param p,q Probability vectors.
#'
#' @return Hellinger distance between \code{p} and \code{q}.
#'
#' @noRd
hellinger_distance <- function(p, q) {
  sqrt(sum((sqrt(p) - sqrt(q))^2)) / sqrt(2)
}


facet_labeller <- function(labels) {
  sapply(labels, function(x) {
    
    # Degree Mixing: DMij → Degree Mixing (i,j)
    if (grepl("^DM\\d{2}$", x)) {
      i <- substr(x, 3, 3)
      j <- substr(x, 4, 4)
      return(paste0("Degree Mixing (", i, ",", j, ")"))
    }
    
    # Mixing: Mij → Mixing (i,j)
    if (grepl("^M\\d{2}$", x)) {
      i <- substr(x, 2, 2)
      j <- substr(x, 3, 3)
      return(paste0("Mixing (", i, ",", j, ")"))
    }
    
    # Degree: degX → Degree X
    if (grepl("^deg\\d+$", x)) {
      num <- sub("deg", "", x)
      return(paste0("Degree ", num))
    }
    
    # Simple replacements
    if (x == "triangles") return("Triangles")
    if (x == "density")   return("Density")
    if (x == "edges")     return("Edges")
    
    # Default: return unchanged
    x
  })
}

.get_distr_settings <- function(distr_name) {
  # Define the core settings for each distribution
  configs <- list(
    "mvn"     = list(sub_code = 1, 
                     mean_bool = TRUE, 
                     var_bool = TRUE, 
                     use_solve_var = TRUE,
                     rules = list(p1 = c("is_numeric"), 
                                  p2 = c("is_numeric", "is_square_mat", "match_matrix_dim"))),
    "normal"     = list(sub_code = 1, 
                     mean_bool = TRUE, 
                     var_bool = TRUE, 
                     use_solve_var = FALSE,
                     rules = list(p1 = c("is_numeric"), 
                                  p2 = c("is_numeric", "all_positive", "match_length"))),
    "lognormal" = list(sub_code = 2, 
                       mean_bool = TRUE, 
                       var_bool = FALSE, 
                       use_solve_var = FALSE,
                       rules = list(p1 = c("is_numeric", "all_positive"))),
    "poisson" = list(sub_code = 3, 
                     mean_bool = TRUE, 
                     var_bool = FALSE, 
                     use_solve_var = FALSE,
                     rules = list(p1 = c("is_numeric", "all_positive"))),
    "uniform" = list(sub_code = 4, 
                     mean_bool = FALSE, 
                     var_bool = FALSE, 
                     use_solve_var = FALSE,
                     rules = list()),
    "beta"     = list(sub_code = 5, 
                        mean_bool = TRUE, 
                        var_bool = TRUE, 
                        use_solve_var = FALSE,
                      rules = list(p1 = c("is_numeric", "all_positive"), 
                                   p2 = c("is_numeric", "all_positive", "match_length"))),
    "dirmult" = list(sub_code = 6, 
                     mean_bool = TRUE, 
                     var_bool = FALSE, 
                     use_solve_var = FALSE,
                     rules = list(p1 = c("is_numeric", "all_positive"))),
    "np" = list(sub_code = 99, 
                     mean_bool = TRUE, 
                     var_bool = FALSE, 
                     use_solve_var = FALSE,
                rules = list(p1 = c("is_numeric", "non_negative", "sums_to_one")))

  )
  
  if (!(distr_name %in% names(configs))) {
    stop(paste("Unsupported distribution:", distr_name))
  }
  
  return(configs[[distr_name]])
}


.VAL_RULES <- list(
  is_numeric    = function(x, name) if (!is.numeric(x)) stop(sprintf("%s must be numeric.", name)),
  all_positive  = function(x, name) if (any(x <= 0 | !is.finite(x))) stop(sprintf("All %s values must be positive.", name)),
  non_negative  = function(x, name) if (any(x < 0 | !is.finite(x))) stop(sprintf("All %s values must be non-negative.", name)),
  sums_to_one   = function(x, name) if (abs(sum(x) - 1) > 1e-8) stop(sprintf("%s must sum to 1.", name)),
  is_square_mat = function(x, name) if (!is.matrix(x) || nrow(x) != ncol(x)) stop(sprintf("%s must be a square matrix.", name)),
  
  # New: Checks if length of p2 matches p1
  match_length = function(x, name, target_val) {
    if (length(x) != length(target_val)) stop(sprintf("Length of %s must match parameter 1.", name))
  },
  
  # New: Checks if MVN matrix rows match mean vector length
  match_matrix_dim = function(x, name, target_val) {
    if (nrow(x) != length(target_val)) stop(sprintf("Dimensions of %s must match length of mean vector.", name))
  }
)

.validate_sample_ccm_inputs <- function(network_stats, prob_distr, prob_distr_params, 
                                        population, sample_size, burnin, interval, 
                                        cov_pattern, initial_g, use_initial_g) {
  


  # 1. Numeric Scalars
  if (!is.numeric(sample_size) || sample_size < 1) stop("sample_size must be >= 1.")
  if (!is.numeric(burnin) || burnin < 1) stop("burnin must be >= 1.")
  if (!is.numeric(interval) || interval < 1) stop("interval must be >= 1.")
  if (!is.numeric(population) || population < 2) stop("population must be >= 2.")
  
  # 2. Covariate Pattern
  if (!is.null(cov_pattern)) {
    if (!is.numeric(cov_pattern) && !is.integer(cov_pattern)) stop("cov_pattern must be numeric.")
    if (length(cov_pattern) != population) {
      stop(sprintf("cov_pattern length (%d) must match population (%d).", 
                   length(cov_pattern), population))
    }
    if (any(is.na(cov_pattern))) stop("cov_pattern cannot contain NAs.")
  }
  
  # 3. Initial Graph
  if (use_initial_g) {
    if (is.null(initial_g) || !inherits(initial_g, "igraph")) {
      stop("When use_initial_g is TRUE, initial_g must be a valid igraph object.")
    }
    if (igraph::vcount(initial_g) != population) {
      stop("initial_g vertex count must match population.")
    }
  }
  
  # 4. Prob Distr Params (Basic structure check)
  if (!is.list(prob_distr_params)) {
    stop("prob_distr_params must be a list.")
  }

  
  # 5. Check Prob Distr Params
  
  if (length(network_stats) != length(prob_distr)) {
    stop(paste0("Mismatched input: 'network_stats' has length ", length(network_stats), 
                ", but 'prob_distr' has length ", length(prob_distr), "."))
  }
  
  for (i in seq_along(prob_distr)) {
    dist_name <- prob_distr[i]
    params    <- prob_distr_params[[i]]
    
    # Fetch settings (including our new rules)
    settings <- .get_distr_settings(dist_name)
    
    # Iterate through the rules for p1, p2, etc.
    for (p_key in names(settings$rules)) {
      p_idx <- as.numeric(gsub("p", "", p_key))
      val   <- params[[p_idx]]
      rules <- settings$rules[[p_key]]
      
      for (rule in rules) {
        # Handle comparison rules that need the first parameter as a target
        if (rule %in% c("match_length", "match_matrix_dim")) {
          .VAL_RULES[[rule]](val, paste(dist_name, p_key), params[[1]])
        } else {
          .VAL_RULES[[rule]](val, paste(dist_name, p_key))
        }
      }
    }
  }
  
  return(invisible(TRUE))
}
