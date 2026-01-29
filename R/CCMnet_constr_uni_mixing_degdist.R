#' Configure CCMnet Constraint Information for Attribute-Mixing and Degree Distribution
#'
#' This function initializes the constraint information for a CCMnet simulation 
#' that simultaneously controls for attribute-specific degree distributions and 
#' attribute-based mixing (nodemix). It supports "Normal" and "Tdist" (t-distribution) 
#' probability models.
#'
#' @param Network_stats A character vector where the first element indicates the 
#'   primary statistic (e.g., "Mixing"). Used to handle parameter ordering.
#' @param Prob_Distr A character vector specifying the distributions. Currently 
#'   supports \code{c("Normal", "Normal")} or \code{c("Tdist", "Tdist")}.
#' @param Prob_Distr_Params A highly nested list. \code{[[1]]} contains degree 
#'   distribution parameters (means and covariances for two attribute groups). 
#'   \code{[[2]]} contains mixing parameters (means and variances for cross-group edges).
#' @param nedges Numeric vector. The first element is the current edge count.
#' @param g A network object (compatible with \code{network} or \code{igraph} 
#'   internal structures) containing nodal attributes.
#' @param max_degree Integer. The maximum degree allowed in the network.
#' @param population Integer. The total number of nodes in the network.
#' @param covPattern A vector of nodal attributes (covariates) used to group nodes 
#'   (e.g., 1 or 2).
#' @param remove_var_last_entry Logical. If \code{TRUE}, removes the final degree 
#'   class from the covariance inversion to avoid singularity in degree distributions.
#'
#' @details 
#' The function calculates current statistics for two distinct groups defined in 
#' \code{covPattern}. It computes:
#' \itemize{
#'   \item Attribute-specific degree distributions using \code{tabulate}.
#'   \item A mixing matrix for edges within group 1, between groups 1 & 2, and within group 2.
#' }
#' For the "Normal" model, it calculates separate precision matrices for the degree 
#' distributions of both groups and concatenates them with the mixing variance.
#' 
#' 
#'
#' @return A list of class \code{CCM_constr_info} containing:
#' \itemize{
#'   \item \code{error}: Integer (0 or 1) indicating if validation failed.
#'   \item \code{prob_type}: Numeric vector (e.g., \code{c(1,1,0,0,1)}) indicating 
#'     the model type for the C-level sampler.
#'   \item \code{mean_vector}: Combined target means for both degree distributions 
#'     and attribute mixing.
#'   \item \code{var_vector}: Concatenated flattened precision matrices.
#'   \item \code{inputs}: A complex numeric vector containing covariate mappings 
#'     and matrix dimensions for C memory management.
#'   \item \code{stats}: Observed counts for edges, degree distributions, and mixing.
#' }
#' 
#' @export

CCMnet_constr_uni_mixing_degdist <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                   nedges, g, max_degree,
                                                   population, covPattern, remove_var_last_entry) {
  
  error = 0
  if (Network_stats[1] == "Mixing") { #swap prob_distr_params
    Prob_Distr_Params_temp = Prob_Distr_Params[[1]]
    Prob_Distr_Params[[1]] = Prob_Distr_Params[[2]]
    Prob_Distr_Params[[2]] = Prob_Distr_Params_temp
  }
  if (length(Prob_Distr_Params[[1]][[1]][[1]]) != length(Prob_Distr_Params[[1]][[1]][[2]])) {
    print("Error: Current limitation requires mean degree distributions to be of equal length.")
    error = 1
  }
  # if (dim(Prob_Distr_Params[[1]][[2]][[1]])[1] != dim(Prob_Distr_Params[[1]][[2]][[2]])[1]) {
  #   print("Error: Current limitation requires covariance matrices to be of equal dimensions.")
  #   error = 1
  # }
  # if (dim(Prob_Distr_Params[[1]][[2]][[1]])[1] != dim(Prob_Distr_Params[[1]][[2]][[1]])[2]) {
  #   print("Error: Covariance matrix is not square.")
  #   error = 1
  # }
  # if (dim(Prob_Distr_Params[[1]][[2]][[2]])[1] != dim(Prob_Distr_Params[[1]][[2]][[2]])[2]) {
  #   print("Error: Covariance matrix is not square.")
  #   error = 1
  # }
  
  if ((Prob_Distr[1] == "Poisson") && ((Prob_Distr[2] == "Poisson"))) {
    covariate_list = covPattern
    
    inputs1 = c(rbind(c(0:(length(Prob_Distr_Params[[1]][[1]][[1]])-1)), rep(1,length(Prob_Distr_Params[[1]][[1]][[1]]))))
    inputs2 = c(rbind(c(0:(length(Prob_Distr_Params[[1]][[1]][[2]])-1)), rep(2,length(Prob_Distr_Params[[1]][[1]][[2]]))))
    
    inputs = c(c(0,1,0,0), length(Prob_Distr_Params[[1]][[1]][[1]]) + length(Prob_Distr_Params[[1]][[1]][[2]]),
               2*(length(Prob_Distr_Params[[1]][[1]][[1]])+length(Prob_Distr_Params[[1]][[1]][[2]])) + population, inputs1, inputs2, covariate_list, c(6,3,6 + population), c(1,2,2,2), covariate_list)
    eta0 = rep(-999.5,length(c(nedges[1], Prob_Distr_Params[[1]][[1]][[1]],Prob_Distr_Params[[1]][[1]][[2]],1,1)))
    
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
    
    stats = c(nedges[1], deg_dist_1, deg_dist_2, mixing[c(2,3)])
    
    mean_vector = c(Prob_Distr_Params[[1]][[1]][[1]], Prob_Distr_Params[[1]][[1]][[2]],  Prob_Distr_Params[[2]][[1]])
    
    var_vector = c(c(0,0),c(0,0), 0,0)
    
    prob_type = c(1,1,0,0,1)
    
  } else if ((Prob_Distr[1] == "Normal") && ((Prob_Distr[2] == "Normal"))) {
    covariate_list = covPattern
    
    # 1. Degree Metadata (16 values)
    inputs_degree_meta = c(rbind(0:3, rep(1, 4)), rbind(0:3, rep(2, 4)))
    
    # 2. Mixing Metadata (6 values)
    inputs_mixing_meta = c(1, 1, 2, 1, 2, 2)
    
    # 3. Build the vector
    inputs = c(
      # Model Header
      c(0, 1, 0, 0), 
      
      # Term 1: Degree (8 stats, 116 total params)
      c(8, 116), 
      inputs_degree_meta, 
      covariate_list, # 100 attributes
      
      # Term 2: Nodemix (3 stats, 106 total params)
      c(6, 3, 106), 
      inputs_mixing_meta, 
      covariate_list # 100 attributes
    )
    
    inputs1 = c(rbind(c(0:(length(Prob_Distr_Params[[1]][[1]][[1]])-1)), rep(1,length(Prob_Distr_Params[[1]][[1]][[1]]))))
    inputs2 = c(rbind(c(0:(length(Prob_Distr_Params[[1]][[1]][[2]])-1)), rep(2,length(Prob_Distr_Params[[1]][[1]][[2]]))))
    
    inputs = c(c(0,1,0,0), length(Prob_Distr_Params[[1]][[1]][[1]]) + length(Prob_Distr_Params[[1]][[1]][[2]]),
               2*(length(Prob_Distr_Params[[1]][[1]][[1]])+length(Prob_Distr_Params[[1]][[1]][[2]])) + population, inputs1, inputs2, covariate_list, c(6,3,6 + population), c(1, 1, 2, 1, 2, 2), covariate_list)
    eta0 = rep(-999.5,length(c(nedges[1], Prob_Distr_Params[[1]][[1]][[1]],Prob_Distr_Params[[1]][[1]][[2]],1,1,1)))
    
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
    
    mean_vector = c(Prob_Distr_Params[[1]][[1]][[1]], Prob_Distr_Params[[1]][[1]][[2]],  Prob_Distr_Params[[2]][[1]])
    
    if (remove_var_last_entry == TRUE) {
      inverse_var_x1 = solve(Prob_Distr_Params[[1]][[2]][[1]][-length(Prob_Distr_Params[[1]][[1]][[1]]),-length(Prob_Distr_Params[[1]][[1]][[1]])])
      inverse_var_x1 = rbind(inverse_var_x1,0)
      inverse_var_x1 = cbind(inverse_var_x1,0)
      
      inverse_var_x2 = solve(Prob_Distr_Params[[1]][[2]][[2]][-length(Prob_Distr_Params[[1]][[1]][[2]]),-length(Prob_Distr_Params[[1]][[1]][[2]])])
      inverse_var_x2 = rbind(inverse_var_x2,0)
      inverse_var_x2 = cbind(inverse_var_x2,0)
    } else {
      inverse_var_x1 = solve(Prob_Distr_Params[[1]][[2]][[1]])
      inverse_var_x2 = solve(Prob_Distr_Params[[1]][[2]][[2]])
    }
    
    var_vector = c(c(inverse_var_x1),c(inverse_var_x2), Prob_Distr_Params[[2]][[2]])
    
    prob_type = c(1,1,0,0,1)
    
  } else if ((Prob_Distr[1] == "Tdist") && ((Prob_Distr[2] == "Tdist"))) {
    if (length(Prob_Distr_Params[[1]]) != 3) {
      
    }
    if (dim(Prob_Distr_Params[[1]][[3]][1]) > 0) {
      print("Error: Degrees of freedom are not greater than 0.")
      error = 1
    }
    if (dim(Prob_Distr_Params[[1]][[3]][2]) > 0) {
      print("Error: Degrees of freedom are not greater than 0.")
      error = 1
    }
    covariate_list = covPattern
    
    inputs1 = c(rbind(c(0:(length(Prob_Distr_Params[[1]][[1]][[1]])-1)), rep(1,length(Prob_Distr_Params[[1]][[1]][[1]]))))
    inputs2 = c(rbind(c(0:(length(Prob_Distr_Params[[1]][[1]][[2]])-1)), rep(2,length(Prob_Distr_Params[[1]][[1]][[2]]))))
    
    inputs = c(c(0,1,0,0), length(Prob_Distr_Params[[1]][[1]][[1]]) + length(Prob_Distr_Params[[1]][[1]][[2]]),
               2*(length(Prob_Distr_Params[[1]][[1]][[1]])+length(Prob_Distr_Params[[1]][[1]][[2]])) + population, inputs1, inputs2, covariate_list, c(4,2,4 + population), c(1,2,2,2), covariate_list)
    eta0 = rep(-999.5,length(c(nedges[1], Prob_Distr_Params[[1]][[1]][[1]],Prob_Distr_Params[[1]][[1]][[2]],1,1)))
    
    mixing = c(0,0,0)
    edge_list = unlist(g$mel)
    dim(edge_list) = c(3,nedges[1])
    for (num_edge in c(1:nedges[1])) {
      if ((covariate_list[edge_list[1,num_edge]] == 1) && (covariate_list[edge_list[2,num_edge]] == 1)) {
        mixing[1] = mixing[1] + 1
      }
      if ((covariate_list[edge_list[1,num_edge]] == 1) && (covariate_list[edge_list[2,num_edge]] == 2)) {
        mixing[2] = mixing[2] + 1
      }
      if ((covariate_list[edge_list[1,num_edge]] == 2) && (covariate_list[edge_list[2,num_edge]] == 1)) {
        mixing[2] = mixing[2] + 1
      }
      if ((covariate_list[edge_list[1,num_edge]] == 2) && (covariate_list[edge_list[2,num_edge]] == 2)) {
        mixing[3] = mixing[3] + 1
      }
    }
    deg_dist_1 = tabulate(degree(g, gmode="graph")[which(covariate_list == 1)]+1)
    deg_dist_2 = tabulate(degree(g, gmode="graph")[which(covariate_list == 2)]+1)
    
    deg_dist_1 = c(tabulate(degree(g, gmode="graph")[which(covariate_list == 1)]+1), rep(0,max(0,length(Prob_Distr_Params[[1]][[1]][[1]])-length(deg_dist_1))))
    deg_dist_2 = c(tabulate(degree(g, gmode="graph")[which(covariate_list == 2)]+1), rep(0,max(0,length(Prob_Distr_Params[[1]][[1]][[2]])-length(deg_dist_2))))
    
    #Assume max degree of both node types is the same
    deg_dist_1 = c(deg_dist_1, rep(0,max(0,length(deg_dist_2)-length(deg_dist_1))))
    deg_dist_2 = c(deg_dist_2, rep(0,max(0,length(deg_dist_1)-length(deg_dist_2))))
    
    stats = c(nedges[1], deg_dist_1, deg_dist_2, mixing[c(2,3)])
    
    mean_vector = c(Prob_Distr_Params[[1]][[1]][[1]], Prob_Distr_Params[[1]][[1]][[2]],  Prob_Distr_Params[[2]][[1]], Prob_Distr_Params[[1]][[3]][1], Prob_Distr_Params[[1]][[3]][2], Prob_Distr_Params[[2]][[3]])
    
    if (remove_var_last_entry == TRUE) {
      inverse_var_x1 = solve(Prob_Distr_Params[[1]][[2]][[1]][-length(Prob_Distr_Params[[1]][[1]][[1]]),-length(Prob_Distr_Params[[1]][[1]][[1]])])
      inverse_var_x1 = rbind(inverse_var_x1,0)
      inverse_var_x1 = cbind(inverse_var_x1,0)
      
      inverse_var_x2 = solve(Prob_Distr_Params[[1]][[2]][[2]][-length(Prob_Distr_Params[[1]][[1]][[2]]),-length(Prob_Distr_Params[[1]][[1]][[2]])])
      inverse_var_x2 = rbind(inverse_var_x2,0)
      inverse_var_x2 = cbind(inverse_var_x2,0)
    } else {
      inverse_var_x1 = solve(Prob_Distr_Params[[1]][[2]][[1]])
      inverse_var_x2 = solve(Prob_Distr_Params[[1]][[2]][[2]])
    }
    
    var_vector = c(c(inverse_var_x1),c(inverse_var_x2), Prob_Distr_Params[[2]][[2]])
    
    prob_type = c(2,2,0,0,1)
    
  } else {
    print("Error: No such distribution for degree distribution and mixing currently implemented.")
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
      Clist_fnamestring = "edges degree_by_attr nodemix",
      Clist_snamestring = "CCMnet CCMnet CCMnet",
      inputs = inputs,
      eta0 = eta0,
      stats = stats,
      MHproposal_name = "TNT",
      MHproposal_package = "CCMnet"
    )
  }
}