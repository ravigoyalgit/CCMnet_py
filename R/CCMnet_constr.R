
CCMnet_constr <- function(Network_stats,
                          Prob_Distr,
                          Prob_Distr_Params, 
                          samplesize,
                          burnin, 
                          interval,
                          statsonly, 
                          P,
                          population, 
                          covPattern) {
  
  CCMnet_Result = CCMnet_constr_py(Network_stats=Network_stats,
                                Prob_Distr=Prob_Distr,
                                Prob_Distr_Params=Prob_Distr_Params, 
                                samplesize = samplesize,
                                burnin=burnin, 
                                interval=interval,
                                statsonly=statsonly, 
                                P=P,
                                population=population, 
                                covPattern = covPattern) 
  
  return(CCMnet_Result)
}
  
  
  