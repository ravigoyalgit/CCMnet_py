#' Calculate initial statistics
#'
#' @noRd

CCMnet_constr_uni_initalstat_mixing_degdist <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
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
  
  deg_dist_1 = tabulate(degree(g)[which(covariate_list == 1)]+1)
  deg_dist_2 = tabulate(degree(g)[which(covariate_list == 2)]+1)
  
  deg_dist_1 = c(tabulate(degree(g)[which(covariate_list == 1)]+1), rep(0,max(0,length(Prob_Distr_Params[[1]][[1]][[1]])-length(deg_dist_1))))
  deg_dist_2 = c(tabulate(degree(g)[which(covariate_list == 2)]+1), rep(0,max(0,length(Prob_Distr_Params[[1]][[1]][[2]])-length(deg_dist_2))))
  
  #Assume max degree of both node types is the same
  deg_dist_1 = c(deg_dist_1, rep(0,max(0,length(deg_dist_2)-length(deg_dist_1))))
  deg_dist_2 = c(deg_dist_2, rep(0,max(0,length(deg_dist_1)-length(deg_dist_2))))
  
  stats = c(nedges[1], deg_dist_1, deg_dist_2, mixing[c(1,2,3)])
  
  CCM_constr_info[["stats"]] <- stats
  return(CCM_constr_info)
}