#' Calculate initial statistics
#'
#' @noRd

CCMnet_constr_uni_initalstat_mixing <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                                           nedges, g, max_degree,
                                           population, covPattern, remove_var_last_entry,
                                     CCM_constr_info) {
  
  u_levels <- sort(unique(covPattern))
  k <- length(u_levels)
  
  # 1. Create an empty k x k matrix
  # Using names helps verify the logic during debugging
  mix_mat <- matrix(0, nrow = k, ncol = k, 
                    dimnames = list(u_levels, u_levels))
  
  # 2. Get the edges
  edge_list <- ends(g, E(g), names = FALSE)
  
  # 3. Fill the full matrix
  for (i in seq_len(nrow(edge_list))) {
    g1 <- as.character(covPattern[edge_list[i, 1]])
    g2 <- as.character(covPattern[edge_list[i, 2]])
    
    # Increment both sides for an undirected relationship
    # This makes the matrix symmetric
    mix_mat[g1, g2] <- mix_mat[g1, g2] + 1
    if (g1 != g2) {
      mix_mat[g2, g1] <- mix_mat[g2, g1] + 1
    }
  }
  
  # 4. Extract the lower triangle (including diagonal)
  # This matches (1,1), (2,1), (2,2), (3,1)... pattern
  mixing <- c()
  for (i in 1:k) {
    for (j in 1:i) {
      mixing <- c(mixing, mix_mat[i, j])
    }
  }
  
  # mixing = c(0,0,0)
  #
  # edge_list <- ends(g, E(g), names = FALSE)
  # for (num_edge in c(1:nedges[1])) {
  #   if ((covariate_list[edge_list[num_edge,1]] == 1) && (covariate_list[edge_list[num_edge,2]] == 1)) {
  #     mixing[1] = mixing[1] + 1
  #   }
  #   if ((covariate_list[edge_list[num_edge,1]] == 1) && (covariate_list[edge_list[num_edge,2]] == 2)) {
  #     mixing[2] = mixing[2] + 1
  #   }
  #   if ((covariate_list[edge_list[num_edge,1]] == 2) && (covariate_list[edge_list[num_edge,2]] == 1)) {
  #     mixing[2] = mixing[2] + 1
  #   }
  #   if ((covariate_list[edge_list[num_edge,1]] == 2) && (covariate_list[edge_list[num_edge,2]] == 2)) {
  #     mixing[3] = mixing[3] + 1
  #   }
  # }
  
  stats = c(nedges[1], mixing)
  
  CCM_constr_info[["stats"]] <- stats
  return(CCM_constr_info)
}
