
import networkx as nx
import pandas as pd
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
  nx.set_node_attributes(g, values = dict(zip(list(range(0, population)), covPattern)) , name = 'covPattern')

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

def save_stats(g_net_stat, results, counter, Network_stats):
  if Network_stats[0] == "Mixing" and len(Network_stats) == 1:
    results[counter] = g_net_stat[np.triu_indices(g_net_stat.shape[0])]

def bayes_inf_MH_prob_calc(MH_prob, g, proposal_edge, P, Ia, Il, R, epi_params):

  if g.has_edge(proposal_edge[0], proposal_edge[1]):  
    p_edge1 = 1 / (1 + math.exp(MH_prob))
  else:
    p_edge1 = math.exp(MH_prob) / (1 + math.exp(MH_prob));

  if g.has_edge(proposal_edge[0], proposal_edge[1]): 
    #Reject the toggle
    MH_prob = log(0)

  else:
    if Ia[proposal_edge[0]] < 999999 or Ia[proposal_edge[1]] < 999999:  

      Il_i = Il[proposal_edge[0]];
      Ia_i = Ia[proposal_edge[0]];
      R_i = R_times[proposal_edge[0]];
      Il_j = Il[proposal_edge[1]];
      Ia_j = Ia[proposal_edge[1]];
      R_j = R_times[proposal_edge[1]];
      
      if (Ia_j < Ia_i):
        time_a = min(Il_j,Ia_i)-Ia_j;
        time_l = max(min(R_j,Ia_i),Il_j) - Il_j;
        muij = math.exp(-beta_a_val*time_a) * math.exp(-beta_l_val*time_l);
      else:
        time_a = min(Il_i,Ia_j)-Ia_i;
        time_l = max(min(R_i,Ia_j),Il_i) - Il_i;
        muij = math.exp(-beta_a_val*time_a) * math.exp(-beta_l_val*time_l);
    
      p_noinfect = (muij*p_edge1)/((1-p_edge1) + muij*p_edge1);
      
      if g.has_edge(proposal_edge[0], proposal_edge[1]): 
        MH_prob = math.log((1-p_noinfect) / p_noinfect)
      else:
        MH_prob = math.log(p_noinfect / (1 - p_noinfect)) 

  return MH_prob 

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
                          bayesian_inference,
                          Ia, 
                          Il, 
                          R, 
                          epi_params,
                          print_calculations):
    
  Prob_Distr_Params = np.array(Prob_Distr_Params)                          
  g = generate_initial_g(population, covPattern)

  if print_calculations:
    print("g info:", nx.info(g))

  g2 = nx.Graph(g)

  g_net_stat = calc_network_stat(g, Network_stats)
  g2_net_stat = np.copy(g_net_stat)

  if print_calculations:
    print("g statistics:", g_net_stat)

  results = [[] for _ in range(samplesize)]
  counter = 0

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

    if bayesian_inference == 1:
      MH_prob = bayes_inf_MH_prob_calc(MH_prob, g, P, Ia, Il, R, epi_params)

    if MH_prob >= 0 or math.log(np.random.uniform(0,1,1)) < MH_prob:
      #Accept proposal
      g = nx.Graph(g2)
      g_net_stat = np.copy(g2_net_stat)
    else:   
      #Reject proposal
      g2 = nx.Graph(g)
      g2_net_stat = np.copy(g_net_stat)

    if (i+1) % interval == 0 and (i+1) > burnin:
      if statsonly:
        save_stats(g_net_stat, results, counter, Network_stats)
      else:
        save_network(results, counter)
      counter = counter + 1

    g_df = nx.to_pandas_edgelist(g)

  results = pd.DataFrame(np.row_stack(results))

  return g_df, results

def R_python_interface_test(Network_stats,
                          Prob_Distr,
                          Prob_Distr_Params, 
                          samplesize,
                          burnin, 
                          interval,
                          statsonly, 
                          P,
                          population, 
                          covPattern,
                          bayesian_inference,
                          Ia, 
                          Il, 
                          R, 
                          epi_params,
                          print_calculations):

  print("Network_stats:", Network_stats, " Type:", type(Network_stats))
  print("Prob_Distr:", Prob_Distr, " Type:", type(Prob_Distr))
  print("Prob_Distr_Params:", Prob_Distr_Params, " Type:", type(Prob_Distr_Params))
  print("samplesize:", samplesize, " Type:", type(samplesize))
  print("burnin:", burnin, " Type:", type(burnin))
  print("interval:", interval, " Type:", type(interval))
  print("statsonly:", statsonly, " Type:", type(statsonly))
  print("P:", P, " Type:", type(P))
  print("population:", population, " Type:", type(population))
  print("covPattern:", covPattern, " Type:", type(covPattern))
  print("bayesian_inference:", bayesian_inference, " Type:", type(bayesian_inference))
  print("Ia:", Ia, " Type:", type(Ia))
  print("Il:", Il, " Type:", type(Il))  
  print("R:", R, " Type:", type(R))
  print("epi_params:", epi_params, " Type:", type(epi_params))
  print("print_calculations:", print_calculations, " Type:", type(print_calculations))

  return(Network_stats)

#################################
#################################
#################################

# Network_stats = ["Mixing"]
# Prob_Distr = ["Multinomial_Poisson"]
# Prob_Distr_Params = [[10], [0.3, 0.4, 0.3]]  

# samplesize = 100
# burnin = 100
# interval = 10
# statsonly = True 
# P = 0
# population = 10
# covPattern = [0,0,0,0,0,1,1,1,1,1]
# bayesian_inference = False
# Ia = [0,0,0,0,0,1,1,1,1,1] 
# Il = [0,0,0,0,0,1,1,1,1,1]
# R = [0,0,0,0,0,1,1,1,1,1]
# epi_params = [0,0,0,0]
# print_calculations = False

# print("Network_stats:", Network_stats, " Type:", type(Network_stats))
# print("Prob_Distr:", Prob_Distr, " Type:", type(Prob_Distr))
# print("Prob_Distr_Params:", Prob_Distr_Params, " Type:", type(Prob_Distr_Params))
# print("samplesize:", samplesize, " Type:", type(samplesize))
# print("burnin:", burnin, " Type:", type(burnin))
# print("interval:", interval, " Type:", type(interval))
# print("statsonly:", statsonly, " Type:", type(statsonly))
# print("P:", P, " Type:", type(P))
# print("population:", population, " Type:", type(population))
# print("covPattern:", covPattern, " Type:", type(covPattern))
# print("bayesian_inference:", bayesian_inference, " Type:", type(bayesian_inference))
# print("Ia:", Ia, " Type:", type(Ia))
# print("Il:", Il, " Type:", type(Il))  
# print("R:", R, " Type:", type(R))
# print("epi_params:", epi_params, " Type:", type(epi_params))
# print("print_calculations:", print_calculations, " Type:", type(print_calculations))

# results = CCMnet_constr_py(Network_stats,
#                           Prob_Distr,
#                           Prob_Distr_Params, 
#                           samplesize,
#                           burnin, 
#                           interval,
#                           statsonly, 
#                           P,
#                           population, 
#                           covPattern,
#                           bayesian_inference,
#                           Ia, 
#                           Il, 
#                           R, 
#                           epi_params,
#                           print_calculations)

# print(results)

# print(type(results))
# print(results)
# print(np.mean(results, axis=0))
