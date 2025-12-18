#' CCM-based Network Inference
#'
#' Fits a null Congruence Class Model (CCM) to an observed network
#' and conducts simulation-based hypothesis tests for additional network
#' properties.
#'
#' @param null_model A formula of the form \code{g ~ edges}, where \code{g} is
#'   an \code{igraph} object.
#' @param alt_model A one-sided formula specifying network properties to test
#'   (e.g., \code{~ degree}). Default is \code{NULL}.
#' @param population Number of nodes in the network.
#' @param Prob_Distr Probability distribution used by \code{CCM_fit}.
#' @param Prob_Distr_Params Parameters for \code{Prob_Distr}.
#' @param covPattern Covariate pattern passed to \code{CCM_fit}.
#' @param samplesize Number of MCMC samples.
#' @param burnin Number of burn-in iterations.
#' @param interval Thinning interval.
#' @param test_stat Test statistic to use. Currently only \code{"hellinger"}.
#'
#' @return An object of class \code{"ccm_inference"}.
#'
#' @export
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
  
  test_stat <- match.arg(test_stat)
  
  parsed <- parse_ccm_formulas(null_model, alt_model)
  
  fit_null <- fit_ccm_null(
    parsed$null_terms,
    population,
    Prob_Distr,
    Prob_Distr_Params,
    covPattern,
    samplesize,
    burnin,
    interval
  )
  
  tests <- run_ccm_tests(
    parsed$g_obs,
    fit_null,
    parsed$alt_terms,
    population,
    test_stat
  )
  
  structure(
    list(
      call = match.call(),
      null_model = null_model,
      alt_model  = alt_model,
      alt_terms  = parsed$alt_terms,
      observed_network = parsed$g_obs,
      fit_null = fit_null,
      tests = tests
    ),
    class = "ccm_inference"
  )
}
