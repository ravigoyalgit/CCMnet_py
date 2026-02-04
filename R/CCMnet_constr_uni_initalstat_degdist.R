#' Calculate initial statistics
#'
#' @noRd

CCMnet_constr_uni_initalstat_degdist <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                                            nedges, g, max_degree,
                                            population, covPattern, remove_var_last_entryy,
                                            CCM_constr_info) {
  
  mean_vector = CCM_constr_info[["mean_vector"]]
  CCM_constr_info[["stats"]] <- c(nedges[1], tabulate(degree(g) + 1),rep(0, length(mean_vector) - length(tabulate(degree(g) + 1))))
  return(CCM_constr_info)
}