#' An internal worker function that performs MCMC sampling for unimodal networks 
#' based on specified network statistics and probability distributions. This 
#' function interfaces with the C-level \code{MCMC_wrapper}.
#'
#' @param Network_stats Character vector. Supported values include "DegreeDist", 
#'   "Density", "Mixing", "DegMixing", and "Triangles".
#' @param Prob_Distr Character string. The distribution type (e.g., "Normal", "NegBin", "DirMult", "NP", "Tdist").
#' @param Prob_Distr_Params List. Distribution parameters (means, covariances, etc.).
#' @param samplesize Integer. Number of network samples to collect.
#' @param burnin Integer. Number of initial MCMC iterations to discard.
#' @param interval Integer. Thinning interval between samples.
#' @param statsonly Logical. If \code{TRUE}, returns statistics; if \code{FALSE}, returns graph objects.
#' @param P An initial \code{igraph} object. If \code{NULL}, a random graph is generated.
#' @param population Integer. The number of nodes in the network.
#' @param covPattern Vector. Categorical nodal attributes for mixing statistics.
#' @param remove_var_last_entry Logical. If \code{TRUE}, the last entry of the variance matrix is dropped for inversion.
#'
#' @return A list containing:
#' \itemize{
#'   \item \code{new_g}: The last sampled \code{igraph} object.
#'   \item \code{statsmatrix}: A matrix of network statistics for each sample.
#' }
#' 
#' @import igraph
#' @keywords internal
#' @export

uni_modal_constr <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                             samplesize, burnin, interval,
                             statsonly, P,
                             population, covPattern, remove_var_last_entry) {
  
  error = 0
  
  if(is.null(covPattern)) {
    covPattern = rep(1,population)
  }
  G_max_degree_bool = FALSE
  ER_prob = .05
  
  if ((length(Network_stats) == 1) && (Network_stats == "DegreeDist")){
    max_degree_f = max_degree = length(Prob_Distr_Params[[1]][[1]])-1
  } else if  ((length(Network_stats) == 1) && (Network_stats == "Density")) {
    max_degree_f = max_degree = population - 1
    if (Prob_Distr == "NP") {
      ER_prob = (max(which(Prob_Distr_Params[[1]][[1]] > 0))-1)/choose(population,2) * .8
    }
  } else if ((length(Network_stats) == 2) && (Network_stats[1] == "DegreeDist") && (Network_stats[2] == "Mixing")) {
    max_degree_1 = length(Prob_Distr_Params[[1]][[1]][[1]])-1
    max_degree_2 = length(Prob_Distr_Params[[1]][[1]][[2]])-1
    max_degree = min(max_degree_1, max_degree_2)
    max_degree_f = max(max_degree_1, max_degree_2)
  } else if ((length(Network_stats) == 2) && (Network_stats[1] == "Mixing") && (Network_stats[2] == "DegreeDist")) {
    max_degree_1 = length(Prob_Distr_Params[[2]][[1]][[1]])-1
    max_degree_2 = length(Prob_Distr_Params[[2]][[1]][[2]])-1
    max_degree = min(max_degree_1, max_degree_2)
    max_degree_f = max(max_degree_1, max_degree_2)
  } else if ((length(Network_stats) == 1) && (Network_stats == "DegMixing")) {
    max_degree_f = max_degree = floor(sqrt(2*length(upper.tri(Prob_Distr_Params[[1]][[1]], diag = TRUE))))
  } else if  ((length(Network_stats) == 2) && (Network_stats[1] == c("DegMixing")) && (Network_stats[2] == c("Triangles"))) {
    max_degree_f = max_degree = floor(sqrt(2*length(upper.tri(Prob_Distr_Params[[1]][[1]], diag = TRUE))))
  } else if  ((length(Network_stats) == 2) && (Network_stats[1] == c("Triangles")) && (Network_stats[2] == c("DegMixing"))) {
    max_degree_f = max_degree = floor(sqrt(2*length(upper.tri(Prob_Distr_Params[[2]][[1]], diag = TRUE))))
  } else {
    max_degree_f = max_degree = population - 1
  }
  
  Gen_Net_counter = 1
  if (is.null(P)) {
    print("Generating Random Initial Network...")
    while (!G_max_degree_bool) {
      if (Gen_Net_counter > 1) {
        print(paste("Try ", Gen_Net_counter, sep=""))
      }
      g <- sample_gnp(n = population, p = ER_prob, directed = FALSE)
      V(g)$CovAttribute <- covPattern
      #g = as.network(rgraph(n=population, m=1, tprob=ER_prob, mode="graph", diag=FALSE, replace=FALSE,
      #       tielist=NULL, return.as.edgelist=FALSE), directed = FALSE)
      #g %v% "CovAttribute" = covPattern
      ER_prob = ER_prob/2
      G_max_degree_bool = max(degree(g)) <= max_degree
      Gen_Net_counter =   Gen_Net_counter + 1
    }
    print("COMPLETED: Generated Random Initial Network")
    P = g
  } else {
    g = P
  }
  
  max_degree = max_degree_f
  
  # --- For network P ---
  # network.edgecount(P) becomes gsize(P)
  Trans_nedges <- c(gsize(P), 0, 0)
  
  # Getting tails and heads: ends() returns a matrix of source and target nodes
  P_edges_matrix <- ends(P, E(P), names = FALSE) 
  Trans_networktails <- P_edges_matrix[, 1]
  Trans_networkheads <- P_edges_matrix[, 2]
  
  # --- For network g ---
  # network.edgecount(g) becomes gsize(g)
  nedges <- c(gsize(g), 0, 0)
  
  # Getting tails and heads for g
  edge_mat_igraph <- ends(g, E(g), names = FALSE)
  tails <- edge_mat_igraph[, 1]
  heads <- edge_mat_igraph[, 2]
  
  # network.size(g) (number of vertices) becomes vcount(g)
  Clist_n <- vcount(g)
  
  # P_edge_mat = unlist(P$mel)
  # dim(P_edge_mat) = c(3,network.edgecount(P))
  # Trans_nedges = c(network.edgecount(P),0,0)
  # Trans_networktails = P_edge_mat[1,]
  # Trans_networkheads = P_edge_mat[2,]
  # 
  # edge_mat = unlist(g$mel)
  # dim(edge_mat) = c(3,network.edgecount(g))
  # 
  # nedges = c(network.edgecount(g),0,0)
  # tails = edge_mat[1,]
  # heads = edge_mat[2,]
  # Clist_n = network.size(g)
  
  if ((length(Network_stats) == 1) && (Network_stats == "DegreeDist")) {
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
      Clist_nterms = 2 #Number of different terms
      Clist_fnamestring = "edges degree"
      Clist_snamestring = "CCMnet CCMnet"
      inputs = c(c(0,1,0,0), length(mean_vector), length(mean_vector), c(0:(length(mean_vector)-1)))
      eta0 = rep(-999.5,length(c(nedges[1], mean_vector,0)))
      stats = c(nedges[1], tabulate(degree(g) + 1),rep(0, length(mean_vector) - length(tabulate(degree(g) + 1))))
      MHproposal_name = "TNT"
      MHproposal_package = "CCMnet"
    }
  } else if ((length(Network_stats) == 1) && (Network_stats == "Density")) {
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
    } else if (Prob_Distr == "NP") {
      prob_type = c(0,0,0,0,99)
      mean_vector = Prob_Distr_Params[[1]][[1]]
      var_vector = c(0,0)
    } else {
      print("Error: No such distribution for degree distribution currently implemented.")
      print("Email ravi.goyal@mail.harvard.edu to add feature.")
      error = 1
    }
    if (error == 0) {
      Clist_nterms = 2 #Number of different terms
      Clist_fnamestring = "edges nfstab"
      Clist_snamestring = "CCMnet CCMnet"
      inputs = c(0,1,0,0,1,0)
      eta0 = c(-999.5, -999.5)
      stats = c(nedges[1],nedges[1])
      MHproposal_name = "TNT"
      MHproposal_package = "CCMnet"
    }
  } else if ((length(Network_stats) == 1) && (Network_stats == "Mixing")) {
    print("Error: No such distribution for mixing currently implemented.")
    print("Email ravi.goyal@mail.harvard.edu to add feature.")
    error = 1
  } else if (((length(Network_stats) == 2) && (Network_stats[1] == "DegreeDist") && (Network_stats[2] == "Mixing")) ||
             ((length(Network_stats) == 2) && (Network_stats[1] == "Mixing") && (Network_stats[2] == "DegreeDist")))  {
    if (Network_stats[1] == "Mixing") { #swap prob_distr_params
      Prob_Distr_Params_temp = Prob_Distr_Params[[1]]
      Prob_Distr_Params[[1]] = Prob_Distr_Params[[2]]
      Prob_Distr_Params[[2]] = Prob_Distr_Params_temp
    }
    if (length(Prob_Distr_Params[[1]][[1]][[1]]) != length(Prob_Distr_Params[[1]][[1]][[2]])) {
      print("Error: Current limitation requires mean degree distributions to be of equal length.")
      error = 1
    }
    if (dim(Prob_Distr_Params[[1]][[2]][[1]])[1] != dim(Prob_Distr_Params[[1]][[2]][[2]])[1]) {
      print("Error: Current limitation requires covariance matrices to be of equal dimensions.")
      error = 1
    }
    if (dim(Prob_Distr_Params[[1]][[2]][[1]])[1] != dim(Prob_Distr_Params[[1]][[2]][[1]])[2]) {
      print("Error: Covariance matrix is not square.")
      error = 1
    }
    if (dim(Prob_Distr_Params[[1]][[2]][[2]])[1] != dim(Prob_Distr_Params[[1]][[2]][[2]])[2]) {
      print("Error: Covariance matrix is not square.")
      error = 1
    }
    if ((Prob_Distr[1] == "Normal") && ((Prob_Distr[2] == "Normal"))) {
      Clist_nterms = 3 #Number of different terms
      Clist_fnamestring = "edges degree_by_attr nodemix"
      Clist_snamestring = "CCMnet CCMnet CCMnet"
      MHproposal_name = "TNT"
      MHproposal_package = "CCMnet"
      covariate_list = covPattern
      
      inputs1 = c(rbind(c(0:(length(Prob_Distr_Params[[1]][[1]][[1]])-1)), rep(1,length(Prob_Distr_Params[[1]][[1]][[1]]))))
      inputs2 = c(rbind(c(0:(length(Prob_Distr_Params[[1]][[1]][[2]])-1)), rep(2,length(Prob_Distr_Params[[1]][[1]][[2]]))))
      
      inputs = c(c(0,1,0,0), length(Prob_Distr_Params[[1]][[1]][[1]]) + length(Prob_Distr_Params[[1]][[1]][[2]]),
                 2*(length(Prob_Distr_Params[[1]][[1]][[1]])+length(Prob_Distr_Params[[1]][[1]][[2]])) + g$gal$n, inputs1, inputs2, covariate_list, c(4,2,4 + g$gal$n), c(1,2,2,2), covariate_list)
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
      Clist_nterms = 3 #Number of different terms
      Clist_fnamestring = "edges degree_by_attr nodemix"
      Clist_snamestring = "CCMnet CCMnet CCMnet"
      MHproposal_name = "TNT"
      MHproposal_package = "CCMnet"
      covariate_list = covPattern
      
      inputs1 = c(rbind(c(0:(length(Prob_Distr_Params[[1]][[1]][[1]])-1)), rep(1,length(Prob_Distr_Params[[1]][[1]][[1]]))))
      inputs2 = c(rbind(c(0:(length(Prob_Distr_Params[[1]][[1]][[2]])-1)), rep(2,length(Prob_Distr_Params[[1]][[1]][[2]]))))
      
      inputs = c(c(0,1,0,0), length(Prob_Distr_Params[[1]][[1]][[1]]) + length(Prob_Distr_Params[[1]][[1]][[2]]),
                 2*(length(Prob_Distr_Params[[1]][[1]][[1]])+length(Prob_Distr_Params[[1]][[1]][[2]])) + g$gal$n, inputs1, inputs2, covariate_list, c(4,2,4 + g$gal$n), c(1,2,2,2), covariate_list)
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
  } else if ((length(Network_stats) == 1) && (Network_stats == "DegMixing")) {
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
    Clist_nterms = 2 #Number of different terms
    Clist_fnamestring = "edges degmix"
    Clist_snamestring = "CCMnet CCMnet"
    MHproposal_name = "TNT"
    MHproposal_package = "CCMnet"
  } else if ((length(Network_stats) == 2) && (Network_stats[1] == c("DegMixing")) && (Network_stats[2] == c("Triangles"))  ||
             (length(Network_stats) == 2) && (Network_stats[1] == c("Triangles")) && (Network_stats[2] == c("DegMixing"))
  ) {
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
    Clist_nterms = 3 #Number of different terms
    Clist_fnamestring = "edges degmix triangle"
    Clist_snamestring = "CCMnet CCMnet CCMnet"
    MHproposal_name = "TNT"
    MHproposal_package = "CCMnet"
    
  } else {
    print("Error: No such distribution for mixing currently implemented.")
    print("Email ravi.goyal@mail.harvard.edu to add feature.")
    error = 1
  }
  
  
  if (error == 0) {
    
    numnetworks = 0 #MCMC_wrapper required
    Clist_dir = FALSE
    Clist_bipartite = FALSE
    maxedges = 200001
    verbose = FALSE
    
    BayesInference = 0 #Required for Bayesian Inference
    TranNet = NULL
    P = P
    Ia = NULL
    Il = NULL
    R_times = NULL
    beta_a = NULL
    beta_l = NULL
    
    NetworkForecast = 0  #Required for Network Forecasting
    evolution_rate_mean = 0
    evolution_rate_var = 0
    
    if (statsonly == FALSE) {
      samplesize_v = rep(1, samplesize)
      burnin_v = c(burnin, rep(interval, samplesize))
      interval_v = rep(interval, samplesize)
    } else {
      samplesize_v = samplesize
      burnin_v = burnin
      interval_v = interval
    }
    
    statsmatrix <- c()
    newnetwork = list()
    
    for (sample_net in c(1:length(samplesize_v))) {
      samplesize = samplesize_v[sample_net]
      burnin = burnin_v[sample_net]
      interval = interval_v[sample_net]
      
      z <- .C("MCMC_wrapper", as.integer(numnetworks),
              as.integer(nedges), as.integer(tails), as.integer(heads),
              as.integer(Clist_n), as.integer(Clist_dir), as.integer(Clist_bipartite),
              as.integer(Clist_nterms), as.character(Clist_fnamestring),
              as.character(Clist_snamestring), as.character(MHproposal_name),
              as.character(MHproposal_package), as.double(inputs), as.double(eta0), as.integer(samplesize),
              s = as.double(rep(stats, samplesize)), as.integer(burnin),
              as.integer(interval), newnwtails = integer(maxedges),
              newnwheads = integer(maxedges), as.integer(verbose),
              as.integer(NULL),
              as.integer(NULL),
              as.integer(NULL),
              as.integer(NULL),
              as.integer(NULL),
              as.integer(FALSE),
              as.integer(0),
              as.integer(maxedges), status = integer(1),
              as.integer(prob_type),   ###MOD ADDED RAVI
              as.integer(max_degree),
              as.double(mean_vector),
              as.double(var_vector),
              as.integer(BayesInference),
              as.integer(Trans_nedges),
              as.integer(Trans_networktails),
              as.integer(Trans_networkheads),
              as.double(Ia),
              as.double(Il),
              as.double(R_times),
              as.double(beta_a),
              as.double(beta_l),
              as.integer(NetworkForecast),
              as.double(evolution_rate_mean),
              as.double(evolution_rate_var),
              as.character(MHproposal_package),
              as.character(MHproposal_package),
              PACKAGE = "CCMnet")
      
      # 1. Extract the number of edges (m) from the first element
      m <- z$newnwtails[1]
      
      # 2. Extract the actual edge IDs (from index 2 to m + 1)
      raw_tails <- z$newnwtails[2:(m + 1)]
      raw_heads <- z$newnwheads[2:(m + 1)]
      
      # 3. Create the edge matrix for igraph
      edges_matrix <- bind_cols(raw_tails, raw_heads)
      
      # 4. Create the new igraph object
      # vertices = nodes_attr_df ensures all original attributes are preserved
      nodes_attr_df = data.frame(name = c(1:(population)), #data.frame(name = c(0:(population-1)), 
                                 covPattern = covPattern)
      
      new_g <- graph_from_data_frame(
        as.data.frame(edges_matrix), 
        directed = FALSE, 
        vertices = nodes_attr_df
      )
      
      statsmatrix <- rbind(statsmatrix, matrix(z$s, nrow = samplesize, ncol = length(stats), byrow = TRUE))
      stats <- statsmatrix[samplesize, ]
      
      # Clean up
      gc()
      
      # nw = g
      # newnetwork[[sample_net]] <- newnw.extract(nw, z, output = "network")
      # 
      # edge_mat = unlist(newnetwork[[sample_net]]$mel)
      # dim(edge_mat) = c(3,network.edgecount(newnetwork[[sample_net]]))
      # 
      # nedges = c(network.edgecount(newnetwork[[sample_net]]),0,0)
      # tails = edge_mat[1,]
      # heads = edge_mat[2,]
      # stats = statsmatrix[sample_net,]
      # 
      # gc()
    }
    
    if (!(is.matrix(statsmatrix))) {
      len_statsmatrix = length(statsmatrix)
      statsmatrix = as.matrix(statsmatrix)
      dim(statsmatrix) = c(1,len_statsmatrix)
    }

    if ((length(Network_stats) == 1) && (Network_stats == "DegreeDist")){
      statsmatrix = statsmatrix[,-1]
      colnames(statsmatrix) = paste("Degree", c(0:(dim(statsmatrix)[2]-1)), sep = " ")
    } else if  ((length(Network_stats) == 1) && (Network_stats == "Density")) {
      statsmatrix = statsmatrix[,1]/choose(population,2)
    } else if ((length(Network_stats) == 2) && (Network_stats[1] == "DegreeDist") && (Network_stats[2] == "Mixing")) {
      statsmatrix = statsmatrix[,-1]
    } else if ((length(Network_stats) == 2) && (Network_stats[1] == "Mixing") && (Network_stats[2] == "DegreeDist")) {
      statsmatrix = statsmatrix[,-1]
    } else if ((length(Network_stats) == 1) && (Network_stats == "DegMixing")) {
      statsmatrix = statsmatrix[,-1]
      colnames(statsmatrix) = paste("Edges", paste(m1, m2, sep = "-"), sep = " ")
    } else if  ((length(Network_stats) == 2) && (Network_stats[1] == c("DegMixing")) && (Network_stats[2] == c("Triangles"))) {
      statsmatrix = statsmatrix[,-1]
      colnames(statsmatrix) = c(paste("Edges", paste(m1, m2, sep = "-"), sep = " "), "Triangles")
    } else if  ((length(Network_stats) == 2) && (Network_stats[1] == c("Triangles")) && (Network_stats[2] == c("DegMixing"))) {
      statsmatrix = statsmatrix[,-1]
      statsmatrix = cbind(statsmatrix[,dim(statsmatrix)[2]], statsmatrix[,-dim(statsmatrix)[2]])
      colnames(statsmatrix) = c("Triangles", paste("Edges", paste(m1, m2, sep = "-"), sep = " "))
    } else {
      statsmatrix = statsmatrix[,-1]
    }
    
    return(list(new_g, as.data.frame(statsmatrix)))
  } else {
    return(list(NULL, NULL))
  }
  
}
