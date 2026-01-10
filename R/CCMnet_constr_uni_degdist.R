#' Configure CCMnet Constraint Information for Degree Distribution Models
#'
#' This function initializes the constraint information required for a CCMnet simulation 
#' when controlling for the network's degree distribution. It handles the 
#' transformation of distribution parameters (e.g., scaling by population) and 
#' calculates the precision matrix for Normal distribution constraints.
#'
#' @param Network_stats A list or vector containing current network statistics.
#' @param Prob_Distr A character string specifying the distribution for the degree 
#'   distribution. Options include "Normal", "NegBin" (Negative Binomial), and 
#'   "DirMult" (Dirichlet-Multinomial).
#' @param Prob_Distr_Params A nested list containing distribution parameters. 
#'   For "Normal", \code{[[1]][[1]]} is the mean vector and \code{[[1]][[2]]} is the 
#'   covariance matrix.
#' @param nedges A numeric vector where the first element is the current edge count.
#' @param g An \code{igraph} object representing the current network.
#' @param max_degree Integer. The maximum degree allowed in the network.
#' @param population Integer. The number of nodes in the network (used for scaling 
#'   means and variances).
#' @param covPattern A vector representing the nodal covariate pattern.
#' @param remove_var_last_entryy Logical. If \code{TRUE}, the last entry of the 
#'   mean vector is excluded from the variance inversion to handle linear 
#'   dependencies in degree distributions.
#'
#' @details 
#' For "Normal" constraints, the function scales the mean vector by \code{1/population} 
#' and the covariance matrix by \code{1/population^2}. It then calculates the 
#' precision matrix using \code{solve()}. If \code{remove_var_last_entryy} is 
#' enabled, it performs a partial inversion and pads the result with zeros.
#'
#' @return A list of class \code{CCM_constr_info} containing configuration for the 
#'   MCMC sampler, including \code{prob_type}, \code{mean_vector}, and the 
#'   flattened precision matrix in \code{var_vector}.
#' 
#' @importFrom igraph degree
#' @export

CCMnet_constr_uni_degdist <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                                            nedges, g, max_degree,
                                            population, covPattern, remove_var_last_entryy) {
  
  error = 0
  if (length(Prob_Distr_Params[[1]][[1]]) < 2) {
    print("Error: length of mean vector is less than 2")
    error = 1
  }
  if (Prob_Distr == "Normal") {
    mean_vector = Prob_Distr_Params[[1]][[1]]
    var_vector = Prob_Distr_Params[[1]][[2]]
    
    mean_vector = mean_vector / population
    var_vector = var_vector / population^2
    prob_type = c(1,0,0,0,1)
    if (dim(var_vector)[1] != dim(var_vector)[2]) {
      print("Error: Covariance matrix is not square")
      error = 1
    }
    if (dim(var_vector)[1] != length(mean_vector)) {
      print("Error: Dimension mismatch between covariance matrix and mean vector")
      error = 1
    }
    
    if (remove_var_last_entry == TRUE) {
      inverse_var_x = solve(var_vector[-length(mean_vector),-length(mean_vector)])
      inverse_var_x = rbind(inverse_var_x,0)
      inverse_var_x = cbind(inverse_var_x,0)
      var_vector = c(inverse_var_x)
    } else {
      var_vector = solve(var_vector)
    }
    
  } else if (Prob_Distr == "NegBin") {
    mean_vector = Prob_Distr_Params[[1]][[1]]
    var_vector = c(0,0)
    prob_type = c(2,0,0,0,1)
  } else if (Prob_Distr == "DirMult") {
    mean_vector = Prob_Distr_Params[[1]][[1]]
    var_vector = c(0,0)
    prob_type = c(3,0,0,0,1)
  } else {
    print("Error: No such distribution for degree distribution currently implemented.")
    print("Email ravi.goyal@mail.harvard.edu to add feature.")
    error = 1
  }
  if (error == 0) {

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
      Clist_fnamestring = "edges degree",
      Clist_snamestring = "CCMnet CCMnet",
      inputs = c(c(0,1,0,0), length(mean_vector), length(mean_vector), c(0:(length(mean_vector)-1))),
      eta0 = rep(-999.5,length(c(nedges[1], mean_vector,0))),
      stats = c(nedges[1], tabulate(degree(g) + 1),rep(0, length(mean_vector) - length(tabulate(degree(g) + 1)))),
      MHproposal_name = "TNT",
      MHproposal_package = "CCMnet"
    )
  }
}