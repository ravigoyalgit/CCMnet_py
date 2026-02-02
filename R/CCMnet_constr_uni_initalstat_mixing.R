#' Calculate initial statistics
#'
#' @keywords internal

CCMnet_constr_uni_initalstat_mixing <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                                           nedges, g, max_degree,
                                           population, covPattern, remove_var_last_entry,
                                     CCM_constr_info) {
  
  covariate_list = covPattern
  
  mixing = c(0,0,0)
  
  edge_list <- ends(g, E(g), names = FALSE)
  for (num_edge in c(1:nedges[1])) {
    if ((covariate_list[edge_list[num_edge,1]] == 1) && (covariate_list[edge_list[num_edge,2]] == 1)) {
      mixing[1] = mixing[1] + 1
    }
    if ((covariate_list[edge_list[num_edge,1]] == 1) && (covariate_list[edge_list[num_edge,2]] == 2)) {
      mixing[2] = mixing[2] + 1
    }
    if ((covariate_list[edge_list[num_edge,1]] == 2) && (covariate_list[edge_list[num_edge,2]] == 1)) {
      mixing[2] = mixing[2] + 1
    }
    if ((covariate_list[edge_list[num_edge,1]] == 2) && (covariate_list[edge_list[num_edge,2]] == 2)) {
      mixing[3] = mixing[3] + 1
    }
  }
  
  stats = c(nedges[1], mixing)
  
  CCM_constr_info[["stats"]] <- stats
  return(CCM_constr_info)
}
