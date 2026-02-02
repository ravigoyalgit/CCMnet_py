#' Fit Null CCM
#'
#' @keywords internal
fit_ccm_null <- function(null_terms,
                         population,
                         Prob_Distr,
                         Prob_Distr_Params,
                         covPattern,
                         samplesize,
                         burnin,
                         interval,
                         alt_terms) {
  
  CCM_fit(
    Network_stats = list(null_terms),
    Prob_Distr = Prob_Distr,
    Prob_Distr_Params = Prob_Distr_Params,
    population = population,
    covPattern = covPattern,
    samplesize = samplesize,
    burnin = burnin,
    interval = interval,
    Obs_stats = list(alt_terms)
  )
}

#' Map Formula Terms to CCM Statistics
#'
#' @keywords internal
stats_from_terms <- function(terms) {
  if ("edges" %in% terms) "Edge" else character(0)
}

#' Hellinger Distance
#'
#' @param p,q Probability vectors.
#'
#' @return Hellinger distance between \code{p} and \code{q}.
#'
#' @keywords internal
hellinger_distance <- function(p, q) {
  sqrt(sum((sqrt(p) - sqrt(q))^2)) / sqrt(2)
}
