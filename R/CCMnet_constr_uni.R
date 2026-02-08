#' An internal worker function that performs MCMC sampling for unimodal networks 
#' based on specified network statistics and probability distributions. This 
#' function interfaces with the C-level \code{MCMC_wrapper}.
#'
#' @param Network_stats Character vector. Supported values include "DegreeDist", 
#'   "edges", "Mixing", "DegMixing", and "Triangles".
#' @param Prob_Distr Character string. The distribution type (e.g., "Normal", "NegBin", "DirMult", "NP", "Tdist").
#' @param Prob_Distr_Params List. Distribution parameters (means, covariances, etc.).
#' @param samplesize Integer. Number of network samples to collect.
#' @param burnin Integer. Number of initial MCMC iterations to discard.
#' @param interval Integer. Thinning interval between samples.
#' @param statsonly Logical. If \code{TRUE}, returns statistics; if \code{FALSE}, returns graph objects.
#' @param G An initial \code{igraph} object. If \code{NULL}, a random graph is generated.
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
#' @noRd

uni_modal_constr <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                             samplesize, burnin, interval,
                             statsonly, G,
                             population, covPattern, remove_var_last_entry,
                             Obs_stats) {
  
  #Verify the inputs for Network_stats, Prob_Distr, and Prob_Distr_Params
  CCM_constr_info = CCMnet_constr_uni_verifyinput(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                  population, covPattern, remove_var_last_entry)
  
  error = CCM_constr_info[["error"]]
  
  if (error == 1) {
    return(list(NULL, NULL))
  }
  
  if(is_empty(covPattern)) {
    covPattern = rep(0L,population)
  }
  ER_prob = .05
  
  if ((length(Network_stats) == 1) && (Network_stats == "DegreeDist")){
    max_degree_f = max_degree = length(Prob_Distr_Params[[1]][[1]])-1
  } else if  ((length(Network_stats) == 1) && (Network_stats == "edges" || Network_stats == "Density")) {
    max_degree_f = max_degree = population - 1
    if (Prob_Distr == "np") {
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
  
  max_degree = max_degree_f
  
  generate_graphs = generate_initial_graph_CCMnet(G, max_degree, ER_prob, covPattern, population)
  P = generate_graphs[[1]]
  g = generate_graphs[[2]]
  
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
  
  #Calculate the initial network statistics for g
  CCM_constr_info = CCMnet_constr_uni_initalstat(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                 nedges, g, max_degree,
                                                 population, covPattern, remove_var_last_entry,
                                                 CCM_constr_info)
  
  if (!(is.null(Obs_stats))) {
    CCM_constr_info = CCMnet_constr_uni_obs_stats(CCM_constr_info, Network_stats, Prob_Distr, Prob_Distr_Params,
                                                  samplesize, burnin, interval,
                                                  statsonly, nedges, g, max_degree,
                                                  population, covPattern, remove_var_last_entry,
                                                  Obs_stats)
  }
  
  error = CCM_constr_info[["error"]]
  prob_type = CCM_constr_info[["prob_type"]]
  mean_vector = CCM_constr_info[["mean_vector"]]
  var_vector = CCM_constr_info[["var_vector"]]
  Clist_nterms = CCM_constr_info[["Clist_nterms"]]
  Clist_fnamestring = CCM_constr_info[["Clist_fnamestring"]]
  Clist_snamestring = CCM_constr_info[["Clist_snamestring"]]
  inputs = CCM_constr_info[["inputs"]]
  eta0 = CCM_constr_info[["eta0"]]
  stats = CCM_constr_info[["stats"]]
  MHproposal_name = CCM_constr_info[["MHproposal_name"]]
  MHproposal_package = CCM_constr_info[["MHproposal_package"]]
  
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
      nodes_attr_df = data.frame(
        name = as.character(1:population), 
        covPattern = covPattern,
        stringsAsFactors = FALSE
      )
      
      edges_df <- as.data.frame(edges_matrix)
      edges_df[[1]] <- as.character(edges_df[[1]])
      edges_df[[2]] <- as.character(edges_df[[2]])
      
      new_g <- graph_from_data_frame(
        d = edges_df, 
        directed = FALSE, 
        vertices = nodes_attr_df
      )
      
      newnetwork[[sample_net]] = new_g
      
      statsmatrix <- rbind(statsmatrix, matrix(z$s, nrow = samplesize, ncol = length(stats), byrow = TRUE))
      stats <- statsmatrix[samplesize, ]
      
      # Clean up
      gc()
      
    }
    
    if (!(is.matrix(statsmatrix))) {
      len_statsmatrix = length(statsmatrix)
      statsmatrix = as.matrix(statsmatrix)
      dim(statsmatrix) = c(1,len_statsmatrix)
    }
    
    ###NEED TO UPDATE######
    if (!(is.null(Obs_stats))) {
        statsmatrix = statsmatrix
    } else {
      if ((length(Network_stats) == 1) && (Network_stats == "DegreeDist")){
        statsmatrix = statsmatrix[,-1]
      } else if  ((length(Network_stats) == 1) && (Network_stats == "edges")) {
        statsmatrix = statsmatrix[,1]
      } else if ((length(Network_stats) == 1)  && (Network_stats == "Density")) {
        statsmatrix = statsmatrix[,1] / choose(population, 2)
      } else if ((length(Network_stats) == 2) && (Network_stats[1] == "DegreeDist") && (Network_stats[2] == "Mixing")) {
        statsmatrix = statsmatrix[,-1]
      } else if ((length(Network_stats) == 2) && (Network_stats[1] == "Mixing") && (Network_stats[2] == "DegreeDist")) {
        statsmatrix = statsmatrix[,-1]
      } else if ((length(Network_stats) == 1) && (Network_stats == "DegMixing")) {
        statsmatrix = statsmatrix[,-1]
      } else if  ((length(Network_stats) == 2) && (Network_stats[1] == c("DegMixing")) && (Network_stats[2] == c("Triangles"))) {
        statsmatrix = statsmatrix[,-1]
      } else if  ((length(Network_stats) == 2) && (Network_stats[1] == c("Triangles")) && (Network_stats[2] == c("DegMixing"))) {
        statsmatrix = statsmatrix[,-1]
        statsmatrix = cbind(statsmatrix[,dim(statsmatrix)[2]], statsmatrix[,-dim(statsmatrix)[2]])
      } else {
        statsmatrix = statsmatrix[,-1]
      } 
    }
    
    return(list(newnetwork, as.data.frame(statsmatrix)))
  } else {
    return(list(NULL, NULL))
  }
  
}
