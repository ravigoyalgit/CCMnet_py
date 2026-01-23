#' Configure CCMnet Constraint Information for Edge-based Models
#'
#' This function initializes the constraint information required for a CCMnet simulation 
#' when controlling for network edges. It maps specified probability distributions 
#' to internal type codes and prepares the parameter vectors for the C-level MCMC routine.
#'
#' @param Network_stats A list or vector containing current network statistics.
#' @param Prob_Distr A character string specifying the distribution for edges. 
#'   Options include "Normal", "LogNormal", "Poisson", "Uniform", and "NP" (Non-Parametric).
#' @param Prob_Distr_Params A nested list containing distribution parameters. 
#'   For "Normal", \code{Prob_Distr_Params[[1]][[1]]} is the mean and \code{[[2]]} is the variance.
#' @param nedges A numeric vector where the first element is the current number of edges in the network.
#' @param g An \code{igraph} object representing the current network.
#' @param max_degree Integer. The maximum degree allowed in the network.
#' @param population Integer. The number of nodes in the network.
#' @param covPattern A vector representing the nodal covariate pattern.
#' @param remove_var_last_entry Logical. Whether to omit the variance for the final entry (used for specific matrix constraints).
#'
#' @return A list of class \code{CCM_constr_info} containing:
#' \itemize{
#'   \item \code{error}: Integer (0 or 1) indicating if the configuration failed.
#'   \item \code{prob_type}: A numeric vector defining the distribution model for C.
#'   \item \code{mean_vector}: Target mean values for the MCMC.
#'   \item \code{var_vector}: Variance values for the MCMC.
#'   \item \code{Clist_nterms}: Number of ERGM terms (default is 2).
#'   \item \code{Clist_fnamestring}: Strings for C-level function mapping.
#'   \item \code{Clist_snamestring}: Strings for package-level function mapping.
#'   \item \code{inputs}: Numeric vector for model-specific constraints.
#'   \item \code{eta0}: Initial natural parameters (usually set very low).
#'   \item \code{stats}: Initial statistics for the model.
#'   \item \code{MHproposal_name}: The name of the Metropolis-Hastings proposal (e.g., "TNT").
#'   \item \code{MHproposal_package}: The package containing the proposal logic.
#' }
#' 
#' @export

CCMnet_constr_uni_edges <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                                    nedges, g, max_degree,
                                    population, covPattern, remove_var_last_entry) {
  
  error = 0
  if (Network_stats == "Edges") {
    if (Prob_Distr == "Normal") {
      prob_type = c(0,0,0,0,1)
      mean_vector = c(Prob_Distr_Params[[1]][[1]],Prob_Distr_Params[[1]][[1]])
      var_vector = c(Prob_Distr_Params[[1]][[2]], Prob_Distr_Params[[1]][[2]])
      if (length(Prob_Distr_Params[[1]][[1]]) != 1) {
        print("Error: mean value for network density is one positive value")
        error = 1
      }
      if (length(Prob_Distr_Params[[1]][[2]]) != 1) {
        print("Error: variance for network density is one positive value")
        error = 1
      }
    } else if (Prob_Distr == "LogNormal") {
      prob_type = c(0,0,0,0,2)
      mean_vector = c(Prob_Distr_Params[[1]][[1]],Prob_Distr_Params[[1]][[1]])
      var_vector = c(0,0)
    } else if (Prob_Distr == "Poisson") {
      prob_type = c(0,0,0,0,3)
      mean_vector = c(Prob_Distr_Params[[1]][[1]],Prob_Distr_Params[[1]][[1]])
      var_vector = c(0,0)
    } else if (Prob_Distr == "Uniform") {
      prob_type = c(0,0,0,0,4)
      mean_vector = c(1, 1)
      var_vector = c(0,0)
    } else if (Prob_Distr == "NP") {
      prob_type = c(0,0,0,0,99)
      mean_vector = Prob_Distr_Params[[1]][[1]]
      var_vector = c(0,0)
    } else {
      print("Error: No such distribution for EDGES currently implemented.")
      error = 1
    } 
  }
  if (Network_stats == "Density") {
    if (Prob_Distr == "Normal") {
      prob_type = c(0,0,0,0,11)
      mean_vector = c(Prob_Distr_Params[[1]][[1]],Prob_Distr_Params[[1]][[1]])
      var_vector = c(Prob_Distr_Params[[1]][[2]], Prob_Distr_Params[[1]][[2]])
      if (length(Prob_Distr_Params[[1]][[1]]) != 1) {
        print("Error: mean value for network density is one positive value")
        error = 1
      }
      if (length(Prob_Distr_Params[[1]][[2]]) != 1) {
        print("Error: variance for network density is one positive value")
        error = 1
      }
    } else if (Prob_Distr == "Beta") {
      prob_type = c(0,0,0,0,12)
      mean_vector = c(Prob_Distr_Params[[1]][[1]],Prob_Distr_Params[[1]][[2]])
      var_vector = c(0,0)
    } else {
      print("Error: No such distribution for DENSITY currently implemented.")
      error = 1
    } 
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
      Clist_fnamestring = "edges nfstab",
      Clist_snamestring = "CCMnet CCMnet",
      inputs = c(0,1,0,0,1,0),
      eta0 = c(-999.5, -999.5),
      stats = c(nedges[1],nedges[1]),
      MHproposal_name = "TNT",
      MHproposal_package = "CCMnet"
    )
  }
  return(CCM_constr_info)
}