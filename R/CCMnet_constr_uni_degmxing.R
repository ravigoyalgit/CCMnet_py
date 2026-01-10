#' Configure CCMnet Constraint Information for Degree Mixing Models
#'
#' This function initializes the constraint information required for a CCMnet simulation 
#' when controlling for the degree mixing patterns of a network. It calculates the 
#' observed degree mixing matrix (DMM) from an initial graph and prepares the 
#' necessary precision matrix and parameter vectors for the MCMC sampler.
#'
#' @param Network_stats A character vector or list containing network statistic names.
#' @param Prob_Distr A character string specifying the distribution for the degree 
#'   mixing statistics. Currently, only "Normal" is implemented.
#' @param Prob_Distr_Params A nested list where \code{[[1]][[1]]} is a numeric 
#'   vector representing the target mean for the upper triangle of the DMM, and 
#'   \code{[[1]][[2]]} is the corresponding covariance matrix.
#' @param nedges Numeric vector. The first element must be the current number of 
#'   edges in the network \code{g}.
#' @param g An \code{igraph} object representing the current network.
#' @param max_degree Integer. The maximum degree to be considered in the mixing 
#'   matrix dimensions.
#' @param population Integer. The total number of nodes in the network.
#' @param covPattern A vector representing the nodal covariate pattern.
#' @param remove_var_last_entry Logical. If \code{TRUE}, the last entry of the 
#'   mean vector is excluded during the covariance inversion to handle linear 
#'   dependencies, and the resulting precision matrix is padded with zeros.
#'
#' @details 
#' The function constructs a \code{max_degree} by \code{max_degree} mixing matrix. 
#' It iterates through the edge list of graph \code{g}, identifying the degrees of 
#' incident nodes and incrementing the corresponding matrix cell. 
#' 
#' For "Normal" constraints, the function calculates the precision matrix by 
#' inverting the provided covariance matrix using \code{solve()}.
#'
#' @return A list of class \code{CCM_constr_info} containing:
#' \itemize{
#'   \item \code{error}: Integer (0 or 1) indicating if the setup failed.
#'   \item \code{prob_type}: Numeric vector \code{c(0,0,1,0,1)} for degree mixing.
#'   \item \code{mean_vector}: Target mean vector for the DMM upper triangle.
#'   \item \code{var_vector}: Flattened precision matrix (inverse covariance).
#'   \item \code{Clist_nterms}: Set to 2 (edges and degree mixing).
#'   \item \code{Clist_fnamestring}: "edges degmix".
#'   \item \code{inputs}: A numeric vector containing matrix indexing metadata for C.
#'   \item \code{stats}: Current observed edges and DMM upper triangle values.
#' }
#' 
#' @importFrom igraph as_edgelist degree
#' @export

CCMnet_constr_uni_degmixing <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                                              nedges, g, max_degree,
                                              population, covPattern, remove_var_last_entry) {
  
  error = 0
  if (Prob_Distr == "Normal") {
    
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
    
    
    m1 = matrix(c(1:max_degree), nrow = max_degree, ncol = max_degree)
    m1 = m1[upper.tri(m1, diag = TRUE)]
    
    m2 = t(matrix(c(1:max_degree), nrow = max_degree, ncol = max_degree))
    m2 = m2[upper.tri(m2, diag = TRUE)]
    
    inputs = c(c(0,1,0), c(((max_degree+1)*max_degree), ((max_degree+1)*max_degree*.5), (((max_degree+1)*max_degree)+1)))
    inputs = c(inputs, m1, m2, max_degree)
    
    eta0 = rep(-999.5,length(c(nedges[1])) + .5*((max_degree+1)*max_degree))
    
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
    
    stats = c(nedges[1],g_dmm[upper.tri(g_dmm, diag = TRUE)] )
    prob_type = c(0,0,1,0,1)
    
    mean_vector = Prob_Distr_Params[[1]][[1]]
    
    if (remove_var_last_entry == TRUE) {
      inverse_var_x = solve(Prob_Distr_Params[[1]][[2]] [-length(mean_vector),-length(mean_vector)])
      inverse_var_x = rbind(inverse_var_x,0)
      inverse_var_x = cbind(inverse_var_x,0)
    } else {
      inverse_var_x = solve(Prob_Distr_Params[[1]][[2]])
    }
    
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
      Clist_nterms = 2, #Number of different terms
      Clist_fnamestring = "edges degmix",
      Clist_snamestring = "CCMnet CCMnet",
      inputs =inputs,
      eta0 = eta0,
      stats = stats,
      MHproposal_name = "TNT",
      MHproposal_package = "CCMnet"
    )
  }
}