#' @importFrom network network.edgecount
#' @noRd

bi_modal_constr <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                            samplesize, burnin, interval,
                            statsonly, P,
                            population, covPattern, remove_var_last_entry,
                            Obs_stats) {
  
  error = 0
  
  if(is.null(covPattern)) {
    covPattern = rep(1,population[1] + population[2])
  }
  
  G_max_degree_bool = FALSE
  ER_prob = .05
  
  if ((length(Network_stats) == 1) && (Network_stats == "DegreeDist")){
    max_degree_1 = length(Prob_Distr_Params[[1]][[1]])-1
    max_degree_2 = length(Prob_Distr_Params[[2]][[1]])-1
    max_degree = min(max_degree_1, max_degree_2)
    max_degree_f = max(max_degree_1, max_degree_2)
    
    len_deg_dist = length(Prob_Distr_Params[[1]][[1]])
    Prob_Distr_Params[[1]][[1]] = as.matrix(Prob_Distr_Params[[1]][[1]])
    dim(Prob_Distr_Params[[1]][[1]]) = c(1, len_deg_dist)
    
    len_deg_dist = length(Prob_Distr_Params[[2]][[1]])
    Prob_Distr_Params[[2]][[1]] = as.matrix(Prob_Distr_Params[[2]][[1]])
    dim(Prob_Distr_Params[[2]][[1]]) = c(1, len_deg_dist)
  } else if ((length(Network_stats) == 2) && (Network_stats[1] == "DegreeDist") && (Network_stats[2] == "Mixing")) {
    max_degree_1 = dim(Prob_Distr_Params[[1]][[1]])[2]-1
    max_degree_2 = dim(Prob_Distr_Params[[2]][[1]])[2]-1
    max_degree = min(max_degree_1, max_degree_2)
    max_degree_f = max(max_degree_1, max_degree_2)
  } else if ((length(Network_stats) == 2) && (Network_stats[1] == "Mixing") && (Network_stats[2] == "DegreeDist")) {
    max_degree_1 = length(Prob_Distr_Params[[1]][[1]])-1
    max_degree_2 = length(Prob_Distr_Params[[2]][[1]])-1
    max_degree = min(max_degree_1, max_degree_2)
    max_degree_f = max(max_degree_1, max_degree_2)
  } else {
    max_degree_f = max_degree = sum(population) - 1
  }
  
  if ((length(Network_stats) == 1) && (Network_stats == "Density")){
    if (is.null(P)) {

      g <-network.initialize(population[1] + population[2],directed = FALSE, bipartite=population[1])
      g %v% "CovAttribute" = covPattern
      
      add.edges(g, tail = 1, head = population[1] + 1)
      
      P = g
    } else {
      g = P
    }
  }
  
  subpopulation = tabulate(covPattern[c(1:population[1])])
  Gen_Net_counter = 1
  if (is.null(P)) {

    g <-network.initialize(population[1] + population[2],directed = FALSE, bipartite=population[1])
    g %v% "CovAttribute" = covPattern
    
    current_node = 1
    for (i in c(1:(dim(Prob_Distr_Params[[1]][[1]])[2]))) { #Go through mean degree distribution - by column
      for (j in c(1:(dim(Prob_Distr_Params[[1]][[1]])[1]))) { #by rows
        num_nodes_deg_i = floor(Prob_Distr_Params[[1]][[1]][j,i]/sum(Prob_Distr_Params[[1]][[1]][j,]) * subpopulation[j]) #i - 1
        if (num_nodes_deg_i > 0) {
          for (k in c(1:num_nodes_deg_i)) { # by number of nodes
            if (i > 1)  {
              if (network.edgecount(g) > 0) {
                valid_population_2 = which(degree(g)[c((population[1]+1):(population[1]+population[2]))] < max_degree)
              } else {
                valid_population_2 = c(1:population[2])
              }
              nodes_type_2 = sample(valid_population_2, size = (i-1), replace = FALSE, prob = NULL)
              add.edges(g, tail = rep(current_node, i-1), head = population[1] + nodes_type_2)
            }
            current_node = current_node + 1
          }
        }
      }
    }
    P = g
  } else {
    g = P
  }
  
  if (is.null(P)) {
    P = g
  }
  
  #max(degree(g))
  
  max_degree = max_degree_f
  
  P_edge_mat = unlist(P$mel)
  dim(P_edge_mat) = c(3,network.edgecount(P))
  Trans_nedges = c(network.edgecount(P),0,0)
  Trans_networktails = P_edge_mat[1,]
  Trans_networkheads = P_edge_mat[2,]
  
  edge_mat = unlist(g$mel)
  dim(edge_mat) = c(3,network.edgecount(g))
  
  nedges = c(network.edgecount(g),0,0)
  tails = edge_mat[1,]
  heads = edge_mat[2,]
  Clist_n = network.size(g)
  
  if ((length(Network_stats) == 1) && (Network_stats == "DegreeDist")) {
    if (length(Prob_Distr_Params[[1]][[1]]) < 2) {
      stop("length of mean vector of type 1 is less than 2")
      error = 1
    }
    if (length(Prob_Distr_Params[[2]][[1]]) < 2) {
      stop("length of mean vector of type 2 is less than 2")
      error = 1
    }
    if (Prob_Distr == "Normal") {
      mean_vector = c(Prob_Distr_Params[[1]][[1]],Prob_Distr_Params[[2]][[1]])
      
      var_vector = c()
      inverse_var_x1 = solve(Prob_Distr_Params[[1]][[2]][-length(Prob_Distr_Params[[1]][[1]]),-length(Prob_Distr_Params[[1]][[1]])])
      inverse_var_x1 = rbind(inverse_var_x1,0)
      inverse_var_x1 = cbind(inverse_var_x1,0)
      
      var_vector = c(var_vector, c(inverse_var_x1))
      
      inverse_var_x2 = solve(Prob_Distr_Params[[2]][[2]][-length(Prob_Distr_Params[[2]][[1]]),-length(Prob_Distr_Params[[2]][[1]])])
      inverse_var_x2 = rbind(inverse_var_x2,0)
      inverse_var_x2 = cbind(inverse_var_x2,0)
      
      var_vector = c(var_vector, c(inverse_var_x2))
      
      prob_type = c(1,0,0,0,1)
      
    } else if (Prob_Distr == "DirMult") {
      mean_vector = c(Prob_Distr_Params[[1]][[1]],Prob_Distr_Params[[2]][[1]])
      var_vector = c(0,0)
      
      prob_type = c(2,0,0,0,1)
    } else if (Prob_Distr == "NP") {
      mean_vector = Prob_Distr_Params[[3]][[1]]
      var_vector = c(0,0)
      
      prob_type = c(99,0,0,0,1)
    } else {
      stop("No such distribution for degree distribution currently implemented.")
      error = 1
    }
    if (error == 0) {
      Clist_nterms = 3 #Number of different terms
      Clist_fnamestring = "edges b1degree b2degree"
      Clist_snamestring = "CCMnet CCMnet CCMnet CCMnet"
      MHproposal_name = "TNT"
      MHproposal_package = "CCMnet"
      
      inputs =  c(c(0,1,0),
                  c(0,length(Prob_Distr_Params[[1]][[1]]),length(Prob_Distr_Params[[1]][[1]]),c(0:(length(Prob_Distr_Params[[1]][[1]])-1))),
                  c(0,length(Prob_Distr_Params[[2]][[1]]),length(Prob_Distr_Params[[2]][[1]]),c(0:(length(Prob_Distr_Params[[2]][[1]])-1)))
      )
      
      stats = c(summary(g ~ edges),
                as.numeric(summary(g ~ b1degree(c(0:max_degree)))),
                as.numeric(summary(g ~ b2degree(c(0:max_degree))))
      )
      
      eta0 = rep(-999.5, length(stats))
    }
  } else if ((length(Network_stats) == 1) && (Network_stats == "Density")) {
    if (Prob_Distr == "Normal") {
      prob_type = c(0,0,0,0,1)
      mean_vector = c(Prob_Distr_Params[[1]][[1]],Prob_Distr_Params[[1]][[1]])
      var_vector = c(Prob_Distr_Params[[1]][[2]], Prob_Distr_Params[[1]][[2]])
      if (length(Prob_Distr_Params[[1]][[1]]) != 1) {
        stop("Error: mean value for network density is one positive value")
        error = 1
      }
      if (length(Prob_Distr_Params[[1]][[2]]) != 1) {
        stop("Error: variance for network density is one positive value")
        error = 1
      }
    } else if (Prob_Distr == "NP") {
      prob_type = c(0,0,0,0,99)
      mean_vector = Prob_Distr_Params[[1]][[1]]
      var_vector = c(0,0)
    } else {
      stop("No such distribution for degree distribution currently implemented.")
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
    stop("No such distribution for mixing currently implemented.")
    error = 1
  } else if (((length(Network_stats) == 2) && (Network_stats[1] == "DegreeDist") && (Network_stats[2] == "Mixing")) ||
             ((length(Network_stats) == 2) && (Network_stats[1] == "Mixing") && (Network_stats[2] == "DegreeDist")))  {
    if (Network_stats[1] == "Mixing") { #swap prob_distr_params
      Prob_Distr_Params_temp = Prob_Distr_Params[[1]]
      Prob_Distr_Params[[1]] = Prob_Distr_Params[[2]]
      Prob_Distr_Params[[2]] = Prob_Distr_Params_temp
      
      Prob_Distr_Params_temp = Prob_Distr_Params[[2]]
      Prob_Distr_Params[[2]] = Prob_Distr_Params[[3]]
      Prob_Distr_Params[[3]] = Prob_Distr_Params_temp
    }
    if ((Prob_Distr[1] == "Normal") && ((Prob_Distr[2] == "Normal"))) {
      
      mean_vector = c(c(t(Prob_Distr_Params[[1]][[1]])),c(t(Prob_Distr_Params[[2]][[1]])),c(t(Prob_Distr_Params[[3]][[1]])))
      
      var_vector = c()
      for (i in c(1:dim(Prob_Distr_Params[[1]][[1]])[1])) {
        inverse_var_x1 = solve(Prob_Distr_Params[[1]][[2]][[i]][-length(Prob_Distr_Params[[1]][[1]][i,]),-length(Prob_Distr_Params[[1]][[1]][i,])])
        inverse_var_x1 = rbind(inverse_var_x1,0)
        inverse_var_x1 = cbind(inverse_var_x1,0)
        
        var_vector = c(var_vector, c(inverse_var_x1))
      }
      
      for (i in c(1:dim(Prob_Distr_Params[[2]][[1]])[1])) {
        inverse_var_x2 = solve(Prob_Distr_Params[[2]][[2]][[i]][-length(Prob_Distr_Params[[2]][[1]][i,]),-length(Prob_Distr_Params[[2]][[1]][i,])])
        inverse_var_x2 = rbind(inverse_var_x2,0)
        inverse_var_x2 = cbind(inverse_var_x2,0)
        
        var_vector = c(var_vector, c(inverse_var_x2))
        
      }
      
      inverse_var_x3 = matrix(0, ncol = length(c(Prob_Distr_Params[[3]][[1]])), nrow =  length(c(Prob_Distr_Params[[3]][[1]])))
      for (i in c(1:dim(Prob_Distr_Params[[3]][[1]])[1])){
        r_index = c((1+(i-1)*dim(Prob_Distr_Params[[3]][[1]])[1]):(i*dim(Prob_Distr_Params[[3]][[1]])[1]))
        inverse_eta_i = Prob_Distr_Params[[3]][[2]][r_index,r_index]
        inverse_eta_i = solve(inverse_eta_i[-dim(Prob_Distr_Params[[3]][[1]])[1],-dim(Prob_Distr_Params[[3]][[1]])[1]])
        inverse_eta_i = rbind(inverse_eta_i,0)
        inverse_eta_i = cbind(inverse_eta_i,0)
        inverse_var_x3[r_index,r_index] = inverse_eta_i
      }
      var_vector = c(var_vector, c(inverse_var_x3))
      
      prob_type = c(1,1,0,0,1)
      
    } else if ((Prob_Distr[1] == "DirMult") && ((Prob_Distr[2] == "DirMult"))) {
      mean_vector = c(c(t(Prob_Distr_Params[[1]][[1]])),c(t(Prob_Distr_Params[[2]][[1]])),c(t(Prob_Distr_Params[[3]][[1]])))
      var_vector = rep(1, dim(Prob_Distr_Params[[1]][[1]])[2]^2 *  dim(Prob_Distr_Params[[1]][[1]])[1] * 2 +  dim(Prob_Distr_Params[[1]][[1]])[1]^4)
      prob_type = c(2,2,0,0,1)
    } else {
      stop("No such distribution for degree distribution and mixing currently implemented.")
      error = 1
    }
    
    if (error == 0) {
      Clist_nterms = 4 #Number of different terms
      Clist_fnamestring = "edges b1degree_by_attr b2degree_by_attr mix"
      Clist_snamestring = "CCMnet CCMnet CCMnet CCMnet"
      MHproposal_name = "TNT"
      MHproposal_package = "CCMnet"
      
      b1deg_by_attr_input = c()
      b2deg_by_attr_input = c()
      for (i in c(1:dim(Prob_Distr_Params[[1]][[1]])[1])) {
        b1deg_by_attr_input = c(b1deg_by_attr_input, c(rbind(c(0:(dim(Prob_Distr_Params[[1]][[1]])[2]-1)),i)))
        b2deg_by_attr_input = c(b2deg_by_attr_input, c(rbind(c(0:(dim(Prob_Distr_Params[[1]][[1]])[2]-1)),i)))
      }
      
      inputs =  c(c(0,1,0),
                  c(0,dim(Prob_Distr_Params[[1]][[1]])[2]*dim(Prob_Distr_Params[[1]][[1]])[1],dim(Prob_Distr_Params[[1]][[1]])[2]*dim(Prob_Distr_Params[[1]][[1]])[1]*2+population[1]),
                  b1deg_by_attr_input, covPattern[c(1:population[1])],
                  c(0,dim(Prob_Distr_Params[[1]][[1]])[2]*dim(Prob_Distr_Params[[1]][[1]])[1],dim(Prob_Distr_Params[[1]][[1]])[2]*dim(Prob_Distr_Params[[1]][[1]])[1]*2+population[2]),
                  b2deg_by_attr_input, covPattern[c((population[1]+1):(population[1] + population[2]))],
                  c(dim(Prob_Distr_Params[[1]][[1]])[1]^2, dim(Prob_Distr_Params[[1]][[1]])[1]^2, dim(Prob_Distr_Params[[1]][[1]])[1]^2*2 + sum(population)),
                  rep(c(1:dim(Prob_Distr_Params[[1]][[1]])[1]), dim(Prob_Distr_Params[[1]][[1]])[1]),
                  rep(c((dim(Prob_Distr_Params[[1]][[1]])[1]+1):(dim(Prob_Distr_Params[[1]][[1]])[1]*2)), each=dim(Prob_Distr_Params[[1]][[1]])[1]),
                  c(covPattern + c(rep(0,population[1]),rep(dim(Prob_Distr_Params[[1]][[1]])[1], population[2]) ))
      )
      
      stats = c(summary(g ~ edges),
                as.numeric(summary(g ~ b1degree(c(0:max_degree), by = "CovAttribute"))),
                as.numeric(summary(g ~ b2degree(c(0:max_degree), by = "CovAttribute"))),
                as.numeric(summary(g ~ nodemix(attrname = "CovAttribute")))
      )
      
      eta0 = rep(-999.5, length(stats))
    }
    
  } else {
    stop("No such Network Statistics currently implemented.")
    error = 1
  }
  
  
  if (error == 0) {
    
    numnetworks = 0 #MCMC_wrapper required
    Clist_dir = FALSE
    Clist_bipartite = population[1]
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
      # Note: If your C code returns 0-based IDs, add 1 for igraph (1-based)
      raw_tails <- z$newnwtails[2:(m + 1)]
      raw_heads <- z$newnwheads[2:(m + 1)]
      
      # 3. Create the edge matrix for igraph
      # cbind creates an m x 2 matrix
      edges_matrix <- cbind(raw_tails, raw_heads)
      
      # 4. Create the new igraph object
      # vertices = nodes_attr_df ensures all original attributes are preserved
      new_g <- graph_from_data_frame(
        as.data.frame(edges_matrix), 
        directed = FALSE, 
        vertices = covPattern
      )
      
      # 5. Store in your list
      newnetwork[[sample_net]] <- new_g
      
      # 6. Prepare 'tails' and 'heads' for the next iteration
      # If you need exactly what you had before (the edge list vectors):
      nedges <- c(gsize(new_g), 0, 0)
      tails  <- raw_tails
      heads  <- raw_heads
      
      # Clean up
      stats <- statsmatrix[sample_net, ]
      gc()
      
      # nw = g
      # statsmatrix <- rbind(statsmatrix, matrix(z$s, nrow = samplesize, ncol = length(stats), byrow = TRUE))
      # newnetwork[[sample_net]] <- newnw.extract(nw, z, output = "network")
      # 
      # edge_mat = unlist(newnetwork[[sample_net]]$mel)
      # dim(edge_mat) = c(3,network.edgecount(newnetwork[[sample_net]]))
      # 
      # nedges = c(network.edgecount(newnetwork[[sample_net]]),0,0)
      # tails = edge_mat[1,]
      # heads = edge_mat[2,]
      # 
      # 
      # stats = statsmatrix[sample_net,]
      # gc()
    }
    
    if (!(is.matrix(statsmatrix))) {
      len_statsmatrix = length(statsmatrix)
      statsmatrix = as.matrix(statsmatrix)
      dim(statsmatrix) = c(1,len_statsmatrix)
    }
    
    if ((length(Network_stats) == 1) && (Network_stats == "DegreeDist")){
      statsmatrix = statsmatrix[,-1]
      colnames(statsmatrix) = c(paste("Type1_Degree", c(0:(dim(statsmatrix)[2]/2-1)), sep = " "),
                                paste("Type1_Degree", c(0:(dim(statsmatrix)[2]/2-1)), sep = " ")
      )
    } else if  ((length(Network_stats) == 1) && (Network_stats == "Density")) {
      statsmatrix[,1] = statsmatrix[,1]/choose(population[1],2)
      statsmatrix[,2] = statsmatrix[,2]/choose(population[2],2)
      statsmatrix = statsmatrix[,c(1:2)]
    } else if ((length(Network_stats) == 2) && (Network_stats[1] == "DegreeDist") && (Network_stats[2] == "Mixing")) {
      statsmatrix = statsmatrix[,-1]
    } else if ((length(Network_stats) == 2) && (Network_stats[1] == "Mixing") && (Network_stats[2] == "DegreeDist")) {
      statsmatrix = statsmatrix[,-1]
    } else {
      statsmatrix = statsmatrix[,-1]
    }
    
    return(list(newnetwork, statsmatrix))
  } else {
    return(list(NULL, NULL))
  }
  
}
