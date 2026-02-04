#' Parse CCM Model Formulas
#'
#' @param null_model Formula specifying the null CCM.
#' @param alt_model Formula specifying alternative terms.
#'
#' @return A list containing the observed network and parsed terms.
#'
#' @noRd
parse_ccm_formulas <- function(null_model, alt_model) {
  
  if (!inherits(null_model, "formula")) {
    stop("null_model must be a formula of the form g ~ edges")
  }
  
  lhs <- null_model[[2]]
  g_obs <- eval(lhs, envir = parent.frame())
  
  if (!inherits(g_obs, "igraph")) {
    stop("Left-hand side of null_model must be an igraph object")
  }
  
  null_terms <- attr(terms(null_model, keep.order = TRUE), "term.labels")
  
  alt_terms <- if (!is.null(alt_model)) {
    attr(terms(alt_model, keep.order = TRUE), "term.labels")
  } else character(0)
  
  list(
    g_obs = g_obs,
    null_terms = null_terms,
    alt_terms = alt_terms
  )
}
