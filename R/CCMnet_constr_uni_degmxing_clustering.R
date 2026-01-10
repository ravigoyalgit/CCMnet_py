#' Configure CCMnet Constraint Information for Degree Mixing and Clustering
#'
#' This function initializes the constraint information required for a CCMnet simulation 
#' that simultaneously controls for degree mixing (the patterns of edges between nodes 
#' of different degrees) and clustering (triangle counts). It prepares the joint 
#' precision matrix and the flattened mixing matrix statistics.
#'
#' @param Network_stats A character vector where the first element indicates the 
#'   primary statistic (e.g., "Triangles"). Used to ensure \code{Prob_Distr_Params} 
#'   are in the correct order.
#' @param Prob_Distr A character vector specifying distributions for mixing and 
#'   clustering (e.g., \code{c("Normal", "Normal")}).
#' @param Prob_Distr_Params A nested list. \code{[[1]]} contains degree mixing 
#'   parameters (mean vector and covariance matrix). \code{[[2]]} contains 
#'   clustering parameters (mean and variance).
#' @param nedges Numeric vector. The first element is the current edge count.
#' @param g An \code{igraph} object representing the current network.
#' @param max_degree Integer. The maximum degree class considered in the mixing matrix.
#' @param population Integer. The total number of nodes in the network.
#' @param covPattern A vector representing nodal covariates.
#' @param remove_var_last_entry Logical. If \code{TRUE}, removes the last entry of 
#'   the mixing matrix from the variance inversion to handle linear dependency.
#'
#' @details 
#' The function constructs a degree mixing matrix (DMM) from the current graph \code{g} 
#' and extracts the upper triangle (including the diagonal). It combines the 
#' DMM statistics with triangle counts calculated via \code{igraph::motifs}.
#' 
#' The precision matrix (\code{var_vector}) is constructed by inverting the 
#' degree mixing covariance and appending the triangle reciprocal variance as 
#' an additional block-diagonal element.
#'
#' @return A list of class \code{CCM_constr_info} containing:
#' \itemize{
#'   \item \code{error}: Integer (0 or 1) indicating if validation failed.
#'   \item \code{prob_type}: Numeric vector \code{c(0,0,1,1,1)} for this model type.
#'   \item \code{mean_vector}: Combined target means for mixing and triangles.
#'   \item \code{var_vector}: Flattened joint precision matrix.
#'   \item \code{stats}: Current observed counts (edges, mixing matrix, triangles).
#'   \item \code{inputs}: Flattened metadata for C-level memory mapping.
#' }
#' 
#' @importFrom igraph as_edgelist degree motifs
#' @export

CCMnet_constr_uni_degmixing_clustering <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                         nedges, g, max_degree,
                                                         population, covPattern, remove_var_last_entry) {
  
  error = 0
  if (Network_stats[1] == "Triangles") { #swap prob_distr_params
    Prob_Distr_Params_temp = Prob_Distr_Params[[1]]
    Prob_Distr_Params[[1]] = Prob_Distr_Params[[2]]
    Prob_Distr_Params[[2]] = Prob_Distr_Params_temp
  }
  if (class(Prob_Distr_Params[[1]][[1]]) != "numeric") {
    print("Error: Mean degree mixing should be a vector representing upper triangle of degree mixing matrix.")
    error = 1
  }
  if (class(Prob_Distr_Params[[1]][[2]])[1] != "matrix") {
    print("Error: Covariance of degree mixing matrix should be a matrix.")
    error = 1
  }
  if (dim(Prob_Distr_Params[[1]][[2]])[1] != dim(Prob_Distr_Params[[1]][[2]])[2]) {
    print("Error: Covariance matrix is not square.")
    error = 1
  }
  if (length(Prob_Distr_Params[[1]][[1]]) != dim(Prob_Distr_Params[[1]][[2]])[2]) {
    print("Error: mean vector and covariance matrix are not similar dimensions.")
    error = 1
  }
  if (length(Prob_Distr_Params[[2]][[1]]) != 1) {
    print("Error: Mean Triangles such be a single positive value.")
    error = 1
  }
  if (length(Prob_Distr_Params[[2]][[2]]) != 1) {
    print("Error: Variance of Triangles such be a single positive value.")
    error = 1
  }
  if ((Prob_Distr[1] == "Normal") && (Prob_Distr[2] == "Normal")) {
    m1 = matrix(c(1:max_degree), nrow = max_degree, ncol = max_degree)
    m1 = m1[upper.tri(m1, diag = TRUE)]
    
    m2 = t(matrix(c(1:max_degree), nrow = max_degree, ncol = max_degree))
    m2 = m2[upper.tri(m2, diag = TRUE)]
    
    inputs = c(c(0,1,0), c(((max_degree+1)*max_degree), ((max_degree+1)*max_degree*.5), (((max_degree+1)*max_degree)+1)))
    inputs = c(inputs, m1, m2, max_degree, c(0,1,0))
    
    eta0 = rep(-999.5,length(c(nedges[1])) + .5*((max_degree+1)*max_degree) + 1)
    
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
    prob_type = c(0,0,1,1,1)
    
    mean_vector = c(Prob_Distr_Params[[1]][[1]], Prob_Distr_Params[[2]][[1]] )
    
    if (remove_var_last_entry == TRUE) {
      inverse_var_x = solve(Prob_Distr_Params[[1]][[2]][-length(mean_vector[-1]),-length(mean_vector[-1])])
      inverse_var_x = rbind(inverse_var_x,0)
      inverse_var_x = cbind(inverse_var_x,0)
    } else {
      inverse_var_x = solve(Prob_Distr_Params[[1]][[2]])
    }
    
    inverse_var_x = rbind(inverse_var_x,0)
    inverse_var_x = cbind(inverse_var_x,0)
    inverse_var_x[dim(inverse_var_x)[1], dim(inverse_var_x)[1]] = 1/Prob_Distr_Params[[2]][[2]]
    
    var_vector = c(inverse_var_x)
  } else {
    print("Error: No such distribution for degree mixing currently implemented.")
    print("Email ravi.goyal@mail.harvard.edu to add feature.")
    error = 1
  }


  if (error == 1) {
    CCM_constr_info <- list(
      error = 1,
      prob_type = NULL,
      mean_vector = NULL,
      var_vector = NULL,
      Clist_nterms = NULL,
      Clist_fnamestring = NULL,
      Clist_snamestring = NULL,
      inputs =  NULL,
      eta0 = NULL,
      stats = NULL,
      MHproposal_name = NULL,
      MHproposal_package = NULL
    )
  }
  if (error == 0) {
    CCM_constr_info <- list(
      error = 0,
      prob_type = prob_type,
      mean_vector = mean_vector,
      var_vector = var_vector,
      Clist_nterms = 3, #Number of different terms
      Clist_fnamestring = "edges degmix triangle",
      Clist_snamestring = "CCMnet CCMnet CCMnet",
      inputs =inputs,
      eta0 = eta0,
      stats = stats,
      MHproposal_name = "TNT",
      MHproposal_package = "CCMnet"
    )
  }

}