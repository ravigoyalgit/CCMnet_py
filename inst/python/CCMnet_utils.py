
import networkx as nx
import pandas as pd
import numpy as np
import random, json
import operator as op
from functools import reduce
from itertools import cycle, islice
import math

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

def R_python_interface_test(Network_stats,
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
                          print_calculations,
                          partial_network,
                          obs_nodes):

  print("Network_stats:", Network_stats, " Type:", type(Network_stats))
  print("Prob_Distr:", Prob_Distr, " Type:", type(Prob_Distr))
  print("Prob_Distr_Params:", Prob_Distr_Params, " Type:", type(Prob_Distr_Params))
  print("Prob_Distr_Params[0]:", Prob_Distr_Params[0], " Type:", type(Prob_Distr_Params[0]))
  #print("Prob_Distr_Params[0][1]:", Prob_Distr_Params[0][1], " Type:", type(Prob_Distr_Params[0][1]))
  #print("Prob_Distr_Params[0][171][4]:", Prob_Distr_Params[0][171][4], " Type:", type(Prob_Distr_Params[0][171][4]))
  #print("Prob_Distr_Params[0][171][3]:", Prob_Distr_Params[0][171][3], " Type:", type(Prob_Distr_Params[0][171][3]))
  #print("Prob_Distr_Params[0][170][4]:", Prob_Distr_Params[0][170][4], " Type:", type(Prob_Distr_Params[0][170][4]))
  #print("Prob_Distr_Params[0][170][3]:", Prob_Distr_Params[0][170][3], " Type:", type(Prob_Distr_Params[0][170][3]))
  print("samplesize:", samplesize, " Type:", type(samplesize))
  print("burnin:", burnin, " Type:", type(burnin))
  print("interval:", interval, " Type:", type(interval))
  print("statsonly:", statsonly, " Type:", type(statsonly))
  print("G:", G, " Type:", type(G))
  print("P:", P, " Type:", type(P))
  print("population:", population, " Type:", type(population))
  print("covPattern:", covPattern, " Type:", type(covPattern))
  print("bayesian_inference:", bayesian_inference, " Type:", type(bayesian_inference))
  print("Ia:", Ia, " Type:", type(Ia))
  print("Il:", Il, " Type:", type(Il))  
  print("R:", R, " Type:", type(R))
  print("epi_params:", epi_params, " Type:", type(epi_params))
  print("print_calculations:", print_calculations, " Type:", type(print_calculations))
  print("partial network:", partial_network, " Type:", type(partial_network))
  print("obs_nodes:", obs_nodes, " Type:", type(obs_nodes))

  return(Network_stats)
