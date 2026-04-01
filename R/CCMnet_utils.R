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
