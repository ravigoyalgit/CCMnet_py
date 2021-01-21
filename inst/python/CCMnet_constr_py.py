
import networkx as nx
import numpy as np
import random
import operator as op
from functools import reduce
import pandas as pd
from itertools import cycle, islice
import math
import scipy.stats as stats

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

def generate_initial_g(population, covPattern):
  
  g = nx.empty_graph(population)
  nx.set_node_attributes(g, values = covPattern, name = 'covPattern')
  return g

def proposal_edge_func(g):

  proposal_edge = random.sample(g.nodes, 2)

  return proposal_edge

def proposal_g2(g, proposal_edge):

  if g.has_edge(proposal_edge[0], proposal_edge[1]):
    #Remove edge
    g.remove_edge(proposal_edge[0], proposal_edge[1])
  else: 
    #Add edge
    g.add_edge(proposal_edge[0], proposal_edge[1])

  return g

def calc_f_mixing(g, g_net_stat, proposal_edge):

  cov0 = g.nodes[proposal_edge[0]]['covPattern']
  cov1 = g.nodes[proposal_edge[1]]['covPattern']

  n_cov0 = sum(x == cov0 for x in nx.get_node_attributes(g, "covPattern").values())
  n_cov1 = sum(x == cov1 for x in nx.get_node_attributes(g, "covPattern").values())

  if g.has_edge(proposal_edge[0], proposal_edge[1]):
    #g->g2 remove edge
      prob_g_g2 = g_net_stat[cov0][cov1]
  else:
    #g->g2 add edge
    if cov0 == cov1:
      prob_g_g2 = ncr(n_cov0,2) - g_net_stat[cov0][cov0]
    else:
      prob_g_g2 = n_cov0 * n_cov1 - g_net_stat[cov0][cov1]

  return prob_g_g2  

def calc_f(g,g2,Network_stats,g_net_stat, g2_net_stat, proposal_edge):
  
  if Network_stats[0] == "Mixing" and len(Network_stats) == 1:
    prob_g_g2 = calc_f_mixing(g,g_net_stat, proposal_edge)

  return prob_g_g2

def calc_network_stat_mixing(g):

  n_cov = len(set(nx.get_node_attributes(g, "covPattern").values()))
  g_net_stat = np.asarray(nx.attr_matrix(g, node_attr="covPattern", normalized=False, rc_order=range(n_cov)))
  return g_net_stat

def calc_network_stat(g, Network_stats):

  if Network_stats[0] == "Mixing" and len(Network_stats) == 1:
    g_net_stat = calc_network_stat_mixing(g)

  return g_net_stat

def calc_network_stat_mixing_2(g, proposal_edge, g_net_stat, g2_net_stat):

  cov0 = g.nodes[proposal_edge[0]]['covPattern']
  cov1 = g.nodes[proposal_edge[1]]['covPattern']

  if g.has_edge(proposal_edge[0], proposal_edge[1]):
    if cov0 == cov1:
      g2_net_stat[cov0][cov1] = g2_net_stat[cov0][cov1] - 1
    else:
      g2_net_stat[cov0][cov1] = g2_net_stat[cov0][cov1] - 1
      g2_net_stat[cov1][cov0] = g2_net_stat[cov1][cov0] - 1
  else:
    if cov0 == cov1:
      g2_net_stat[cov0][cov1] = g2_net_stat[cov0][cov1] + 1
    else:
      g2_net_stat[cov0][cov1] = g2_net_stat[cov0][cov1] + 1
      g2_net_stat[cov1][cov0] = g2_net_stat[cov1][cov0] + 1

  return g2_net_stat

def calc_network_stat_2(g, Network_stats, proposal_edge, g_net_stat, g2_net_stat):

  if Network_stats[0] == "Mixing" and len(Network_stats) == 1:
    g2_net_stat = calc_network_stat_mixing_2(g, proposal_edge, g_net_stat, g2_net_stat)

  return g2_net_stat


def calc_prob_mixing(g_net_stat, Prob_Distr, Prob_Distr_Params):

  if Prob_Distr[0] == "Multinomial_Poisson":
    x1 = sum(g_net_stat[np.triu_indices(g_net_stat.shape[0])])
    log_prob_x1 = stats.poisson.logpmf(x1, Prob_Distr_Params[0])

    log_prob_x2 = stats.multinomial.logpmf(g_net_stat[np.triu_indices(g_net_stat.shape[0])], n=x1, p=Prob_Distr_Params[1])

    prob_g = log_prob_x1 + log_prob_x2

  return prob_g

def calc_prob(g_net_stat, Network_stats, Prob_Distr, Prob_Distr_Params):

  if Network_stats[0] == "Mixing" and len(Network_stats) == 1:
    prob_g = calc_prob_mixing(g_net_stat, Prob_Distr, Prob_Distr_Params)

  return prob_g

def CCMnet_constr_py(Network_stats,
                          Prob_Distr,
                          Prob_Distr_Params, 
                          samplesize,
                          burnin, 
                          interval,
                          statsonly, 
                          P,
                          population, 
                          covPattern,
                          print_calculations):
                            
  g = generate_initial_g(population, covPattern)
  g2 = nx.Graph(g)

  g_net_stat = calc_network_stat(g, Network_stats)
  g2_net_stat = np.copy(g_net_stat)

  if print_calculations:
    print("g:", g_net_stat)

  for i in range(burnin+samplesize*interval):
    proposal_edge = proposal_edge_func(g)
    g2 = proposal_g2(g2, proposal_edge)

    g2_net_stat = calc_network_stat_2(g, Network_stats, proposal_edge, g_net_stat, g2_net_stat)

    f_g_g2 = calc_f(g,g2,Network_stats, g_net_stat, g2_net_stat, proposal_edge)
    f_g2_g = calc_f(g2,g,Network_stats, g2_net_stat, g_net_stat, proposal_edge)

    prob_g = calc_prob(g_net_stat, Network_stats, Prob_Distr, Prob_Distr_Params)
    prob_g2 = calc_prob(g2_net_stat, Network_stats, Prob_Distr, Prob_Distr_Params)

    if print_calculations:
      print("g############")
      print(g_net_stat)
      print(f_g_g2)
      print(prob_g)

      print("g2############")
      print(g2_net_stat)
      print(f_g2_g)
      print(prob_g2)

    if math.isnan(prob_g):
      MH_prob = 1
    elif math.isnan(prob_g2):
      MH_prob = 0
    else: 
      MH_prob = math.log(f_g2_g) + prob_g2 - (math.log(f_g_g2) + prob_g)
    
    if print_calculations:
      print(MH_prob)

    if MH_prob > np.random.uniform(0,1,1):
      #Accept proposal
      g = nx.Graph(g2)
      g_net_stat = np.copy(g2_net_stat)
    else:   
      #Reject proposal
      g2 = nx.Graph(g)
      g2_net_stat = np.copy(g_net_stat)

    if statsonly:
      save_stats()
    else:
      save_network()

  return g

#################################
#################################
#################################

Network_stats = ["Mixing"]
Prob_Distr = ["Multinomial_Poisson"]
Prob_Distr_Params = np.array([[4], [0.3, 0.4, 0.3]])  
samplesize = 1
burnin = 2
interval = 1
statsonly = True 
P = 0
population = 10
covPattern_keys = list(range(0, population))
covPattern_values = [0,0,0,0,0,1,1,1,1,1]
covPattern = dict(zip(covPattern_keys, covPattern_values)) 
print_calculations = False

g = CCMnet_constr_py(Network_stats,
                          Prob_Distr,
                          Prob_Distr_Params, 
                          samplesize,
                          burnin, 
                          interval,
                          statsonly, 
                          P,
                          population, 
                          covPattern,
                          print_calculations)
