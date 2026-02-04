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
