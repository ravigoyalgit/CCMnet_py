
CCMnet_constr <- function(Network_stats,
                          Prob_Distr,
                          Prob_Distr_Params, 
                          samplesize,
                          burnin, 
                          interval,
                          statsonly,
                          G,
                          P,
                          population, 
                          covPattern,
                          bayesian_inference,
                          Ia, 
                          Il, 
                          R, 
                          epi_params,
                          print_calculations) {
  
  samplesize = as.integer(samplesize)
  burnin = as.integer(burnin)
  interval = as.integer(interval)
  population = as.integer(population)
  covPattern = as.integer(covPattern)
  
  results = CCMnet_constr_py(Network_stats,
                   Prob_Distr,
                   Prob_Distr_Params, 
                   samplesize,
                   burnin, 
                   interval,
                   statsonly,
                   G,
                   P,
                   population, 
                   covPattern,
                   bayesian_inference,
                   Ia, 
                   Il, 
                   R, 
                   epi_params,
                   print_calculations)
  
  nodes_attr_df = data.frame(name = c(0:(population-1)), 
                             covPattern = covPattern)
  g = graph_from_data_frame(results[[1]], directed=FALSE, vertices = nodes_attr_df)

  return(list(g, results[[2]]))
}
  
  
  