#' Calculate initial statistics
#'
#' @noRd

CCMnet_constr_uni_initalstat_edges <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                                    nedges, g, max_degree,
                                    population, covPattern, remove_var_last_entry,
                                    CCM_constr_info) {
  
  CCM_constr_info[["stats"]] <- c(nedges[1],nedges[1])
  return(CCM_constr_info)
}