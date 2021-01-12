
def generate_initial_g(population, covPattern):
  
  g = nx.Graph()
  return g

def CCMnet_constr_py(Network_stats,
                          Prob_Distr,
                          Prob_Distr_Params, 
                          samplesize,
                          burnin, 
                          interval,
                          statsonly, 
                          P,
                          population, 
                          covPattern):
                            
  g = generate_initial_g(population, covPattern)
  
  return g
