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

.get_supported_distrs <- function(stat_key) {
  distrs <- list(
    "edges"               = c("poisson", "uniform", "np"),
    "density"             = c("normal", "beta"),
    "degreedist"          = c("dirmult"),
    "degmixing"           = c("mvn"),
    "mixing"              = c("poisson"),
    "degmixing+triangles" = c("mvn+normal"),
    "degreedist+mixing"   = c("mvn+normal")
  )
  
  if (!(stat_key %in% names(distrs))) return(character(0))
  return(distrs[[stat_key]])
}


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
  
  return(invisible(TRUE))
}
