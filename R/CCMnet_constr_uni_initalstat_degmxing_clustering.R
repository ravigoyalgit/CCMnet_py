#' Calculate initial statistics
#'
#' @noRd

CCMnet_constr_uni_initalstat_degmixing_clustering <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                         nedges, g, max_degree,
                                                         population, covPattern, remove_var_last_entry,
                                                   CCM_constr_info) {
  
  g_dmm = matrix(0,  nrow = max_degree, ncol = max_degree)
  edge_list = as_edgelist(g)
  g_degree = degree(g)
  for (num_edge in c(1:nedges[1])) {
    deg1 = g_degree[edge_list[num_edge,1]]
    deg2 = g_degree[edge_list[num_edge,2]]
    if ((deg1 <= max_degree) && (deg2 <= max_degree)) {
      g_dmm[deg1, deg2] = g_dmm[deg1,deg2] + 1
      if (deg1 != deg2) {
        g_dmm[deg2, deg1] = g_dmm[deg2,deg1] + 1
      }
    }
  }
  
  stats = c(nedges[1],g_dmm[upper.tri(g_dmm, diag = TRUE)], motifs(g, size = 3)[4])
  CCM_constr_info[["stats"]] <- stats
  return(CCM_constr_info)
}