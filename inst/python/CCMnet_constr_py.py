
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
import cProfile

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

def calc_f_mixing(g_net_stat, proposal_edge, g_proposal_edge, covPattern, bayesian_inference, P_net_stat):

  cov0 = covPattern[proposal_edge[0]]
  cov1 = covPattern[proposal_edge[1]]

  n_cov0 = sum(x == cov0 for x in covPattern)
  n_cov1 = sum(x == cov1 for x in covPattern)

  if g_proposal_edge:
    #g->g2 remove edge
    prob_g_g2 = g_net_stat[cov0][cov1]
    if bayesian_inference:
      prob_g_g2 = prob_g_g2 #- P_net_stat[cov0][cov1]
  else:
    #g->g2 add edge
    if cov0 == cov1:
      prob_g_g2 = ncr(n_cov0,2) - g_net_stat[cov0][cov0]
    else:
      prob_g_g2 = n_cov0 * n_cov1 - g_net_stat[cov0][cov1]

  return prob_g_g2  

def calc_f_degree(g_net_stat, proposal_edge, g_proposal_edge, covPattern, bayesian_inference, P_net_stat, g, f_g_g2_bool):

  if f_g_g2_bool:
    g_degree_0 = g.degree[proposal_edge[0]]
    g_degree_1 = g.degree[proposal_edge[1]]
  else:
    if g_proposal_edge:
      g_degree_0 = g.degree[proposal_edge[0]] + 1
      g_degree_1 = g.degree[proposal_edge[1]] + 1
    else:
      g_degree_0 = g.degree[proposal_edge[0]] - 1
      g_degree_1 = g.degree[proposal_edge[1]] - 1     


  g_Deg_Distr_Edges = np.multiply(g_net_stat,range(len(g_net_stat)))

  g_ecount = sum(g_Deg_Distr_Edges)/2

  if g_ecount != 0:
      g_exp_dmm = (g_Deg_Distr_Edges[g_degree_0] * g_Deg_Distr_Edges[g_degree_1])/ (2.0*g_ecount)
      if g_degree_0 == g_degree_1:
          g_exp_dmm = g_exp_dmm * .5
  else:
      g_exp_dmm = 0
  
  if g_proposal_edge:
    #g->g2 remove edge
    prob_g_g2 = g_exp_dmm
  else:
    #g->g2 add edge
    if g_degree_0 == g_degree_1:
      prob_g_g2 = (g_net_stat[g_degree_0]* (g_net_stat[g_degree_0]-1)*.5) - g_exp_dmm
    else:
      prob_g_g2 = g_net_stat[g_degree_0]* g_net_stat[g_degree_1] - g_exp_dmm
      
  return prob_g_g2  

def calc_f(Network_stats,g_net_stat, g2_net_stat, proposal_edge, g_proposal_edge, covPattern, bayesian_inference, P_net_stat, g, f_g_g2_bool):
  
  if Network_stats[0] == "Mixing" and len(Network_stats) == 1:
    prob_g_g2 = calc_f_mixing(g_net_stat, proposal_edge, g_proposal_edge, covPattern, bayesian_inference, P_net_stat)

  if Network_stats[0] == "Degree" and len(Network_stats) == 1:
    prob_g_g2 = calc_f_degree(g_net_stat, proposal_edge, g_proposal_edge, covPattern, bayesian_inference, P_net_stat, g, f_g_g2_bool)

  return prob_g_g2

def calc_network_stat_mixing(g):

  n_cov = len(set(nx.get_node_attributes(g, "covPattern").values()))
  g_net_stat = np.asarray(nx.attr_matrix(g, node_attr="covPattern", normalized=False, rc_order=range(n_cov)))
  return g_net_stat

def calc_network_stat_degree(g):

  g_net_stat = nx.degree_histogram(g)
  g_net_stat.extend([0] * (g.number_of_nodes() - len(g_net_stat)))

  return  g_net_stat

def calc_network_stat(g, Network_stats):

  if Network_stats[0] == "Mixing" and len(Network_stats) == 1:
    g_net_stat = calc_network_stat_mixing(g)

  if Network_stats[0] == "Degree" and len(Network_stats) == 1:
    g_net_stat = calc_network_stat_degree(g)

  return g_net_stat

def calc_network_stat_mixing_2(proposal_edge, g_net_stat, g2_net_stat, g_proposal_edge, covPattern):

  cov0 = covPattern[proposal_edge[0]]
  cov1 = covPattern[proposal_edge[1]]

  if g_proposal_edge:
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

def calc_network_stat_degree_2(proposal_edge, g_net_stat, g2_net_stat, g_proposal_edge, covPattern, g):

  deg0 = g.degree[proposal_edge[0]]
  deg1 = g.degree[proposal_edge[1]]

  if g_proposal_edge:
    g2_net_stat[deg0] = g2_net_stat[deg0] - 1
    g2_net_stat[deg1] = g2_net_stat[deg1] - 1
    g2_net_stat[deg0-1] = g2_net_stat[deg0-1] + 1
    g2_net_stat[deg1-1] = g2_net_stat[deg1-1] + 1
  else:
    g2_net_stat[deg0] = g2_net_stat[deg0] - 1
    g2_net_stat[deg1] = g2_net_stat[deg1] - 1
    g2_net_stat[deg0+1] = g2_net_stat[deg0+1] + 1
    g2_net_stat[deg1+1] = g2_net_stat[deg1+1] + 1

  return g2_net_stat 

def calc_network_stat_2(Network_stats, proposal_edge, g_net_stat, g2_net_stat, g_proposal_edge, covPattern, g):

  if Network_stats[0] == "Mixing" and len(Network_stats) == 1:
    g2_net_stat = calc_network_stat_mixing_2(proposal_edge, g_net_stat, g2_net_stat, g_proposal_edge, covPattern)

  if Network_stats[0] == "Degree" and len(Network_stats) == 1:
    g2_net_stat = calc_network_stat_degree_2(proposal_edge, g_net_stat, g2_net_stat, g_proposal_edge, covPattern, g)

  return g2_net_stat

def calc_probs_mixing(g_net_stat, g2_net_stat, proposal_edge, covPattern, Prob_Distr, Prob_Distr_Params):

  x_g = sum(g_net_stat[np.triu_indices(g_net_stat.shape[0])])
  x_g2 = sum(g2_net_stat[np.triu_indices(g2_net_stat.shape[0])])

  cov0 = covPattern[proposal_edge[0]]
  cov1 = covPattern[proposal_edge[1]]

  entry_id = sum(range(cov1+1)) + cov0

  if x_g > x_g2:
    #removed edge
    log_prob_x_g2 = -math.log(Prob_Distr_Params[0]) + math.log(x_g)
    log_prob_x2_g2 = -math.log(x_g) + math.log(g_net_stat[cov0][cov1]) - math.log(Prob_Distr_Params[1][entry_id])
  else:
    log_prob_x_g2 = math.log(Prob_Distr_Params[0]) - math.log(x_g2)
    log_prob_x2_g2 = math.log(x_g2) - math.log(g2_net_stat[cov0][cov1]) + math.log(Prob_Distr_Params[1][entry_id])

  prob_g = 0
  prob_g2 = log_prob_x_g2 + log_prob_x2_g2

  return prob_g, prob_g2

def calc_probs_degree(g_net_stat, g2_net_stat, proposal_edge, covPattern, Prob_Distr, Prob_Distr_Params, g, g_proposal_edge):

  deg0 = g.degree[proposal_edge[0]]
  deg1 = g.degree[proposal_edge[1]]

  if g_proposal_edge:
    #removed edge
    if (deg0 == deg1):
      log_prob_g2 = -math.log(g2_net_stat[deg0-1]) - math.log(g2_net_stat[deg0-1]-1) + math.log(g_net_stat[deg0]) + math.log(g_net_stat[deg0]-1) + 2*math.log(Prob_Distr_Params[1][deg0-1]) - 2*math.log(Prob_Distr_Params[1][deg0])
    elif (deg0 == deg1 + 1):
      log_prob_g2 = -math.log(g2_net_stat[deg1-1]) + math.log(g_net_stat[deg0]) + math.log(Prob_Distr_Params[1][deg1-1]) - math.log(Prob_Distr_Params[1][deg0])
    elif (deg0 == deg1 - 1):
      log_prob_g2 = -math.log(g2_net_stat[deg0-1]) + math.log(g_net_stat[deg1]) + math.log(Prob_Distr_Params[1][deg0-1]) - math.log(Prob_Distr_Params[1][deg1])
    else:
      log_prob_g2 = -math.log(g2_net_stat[deg0-1]) - math.log(g2_net_stat[deg1-1]) + math.log(g_net_stat[deg0]) + math.log(g_net_stat[deg1]) + math.log(Prob_Distr_Params[1][deg0-1]) + math.log(Prob_Distr_Params[1][deg1-1]) - math.log(Prob_Distr_Params[1][deg0]) - math.log(Prob_Distr_Params[1][deg1])
  else:
    #add edge
    if (deg0 == deg1):
      log_prob_g2 = math.log(g_net_stat[deg0]) + math.log(g_net_stat[deg0]-1) - math.log(g2_net_stat[deg0+1]) - math.log(g2_net_stat[deg0+1]-1) - 2*math.log(Prob_Distr_Params[1][deg0]) + 2*math.log(Prob_Distr_Params[1][deg0+1])
    elif (deg0 == deg1 + 1):
      log_prob_g2 = math.log(g_net_stat[deg1]) - math.log(g2_net_stat[deg0+1]) - math.log(Prob_Distr_Params[1][deg1]) + math.log(Prob_Distr_Params[1][deg0+1])
    elif (deg0 == deg1 - 1):
      log_prob_g2 = math.log(g_net_stat[deg0]) - math.log(g2_net_stat[deg1+1]) - math.log(Prob_Distr_Params[1][deg0]) + math.log(Prob_Distr_Params[1][deg1+1])
    else:
      log_prob_g2 = math.log(g_net_stat[deg0]) + math.log(g_net_stat[deg1]) - math.log(g2_net_stat[deg0+1]) - math.log(g2_net_stat[deg1+1]) - math.log(Prob_Distr_Params[1][deg0]) - math.log(Prob_Distr_Params[1][deg1]) + math.log(Prob_Distr_Params[1][deg0+1]) + math.log(Prob_Distr_Params[1][deg1+1])

  prob_g = 0
  prob_g2 = log_prob_g2

  return prob_g, prob_g2

def calc_probs(g_net_stat, g2_net_stat, proposal_edge, covPattern, Network_stats, Prob_Distr, Prob_Distr_Params, g, g_proposal_edge):

  if Network_stats[0] == "Mixing" and len(Network_stats) == 1:
    prob_g, prob_g2 = calc_probs_mixing(g_net_stat, g2_net_stat, proposal_edge, covPattern, Prob_Distr, Prob_Distr_Params)

  if Network_stats[0] == "Degree" and len(Network_stats) == 1:
    prob_g, prob_g2 = calc_probs_degree(g_net_stat, g2_net_stat, proposal_edge, covPattern, Prob_Distr, Prob_Distr_Params, g, g_proposal_edge)

  return prob_g, prob_g2

def save_stats(g_net_stat, results, counter, Network_stats):
  if Network_stats[0] == "Mixing" and len(Network_stats) == 1:
    results[counter] = g_net_stat[np.triu_indices(g_net_stat.shape[0])]
  if Network_stats[0] == "Degree" and len(Network_stats) == 1:
    results[counter] = g_net_stat

def bayes_inf_MH_prob_calc(MH_prob, g, proposal_edge, Pnet, Ia, Il, R, epi_params):

  beta_a_val = epi_params[0]
  beta_l_val = epi_params[1]

  if g.has_edge(proposal_edge[0], proposal_edge[1]):  
    p_edge1 = 1 / (1 + math.exp(MH_prob))
  else:
    p_edge1 = math.exp(MH_prob) / (1 + math.exp(MH_prob));

  if Pnet.has_edge(proposal_edge[0], proposal_edge[1]): 
    #Reject the toggle
    MH_prob = -math.inf
  else:
    if Ia[proposal_edge[0]] < 999999 or Ia[proposal_edge[1]] < 999999:  

      Il_i = Il[proposal_edge[0]];
      Ia_i = Ia[proposal_edge[0]];
      R_i = R[proposal_edge[0]];
      Il_j = Il[proposal_edge[1]];
      Ia_j = Ia[proposal_edge[1]];
      R_j = R[proposal_edge[1]];
      
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

def generate_net(population, P, covPattern):
  Pnet = nx.empty_graph(population)
  Pnet.add_edges_from(P)
  nx.set_node_attributes(Pnet, values = dict(zip(list(range(0, population)), covPattern)) , name = 'covPattern')

  return Pnet

def CCMnet_constr_py(Network_stats,
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
                          print_calculations):
    
  Prob_Distr_Params = np.array(Prob_Distr_Params)                          

  if bayesian_inference == 1:
    G_list = [tuple(r) for r in G.to_numpy().tolist()]
    g = generate_net(population, G_list, covPattern)
  else:
    g = generate_initial_g(population, covPattern)

  g_net_stat = calc_network_stat(g, Network_stats)
  g2_net_stat = np.copy(g_net_stat)

  if print_calculations:
    print("g info:", nx.info(g))
    print("g statistics:", g_net_stat)

  if bayesian_inference == 1:
    P_list = [tuple(r) for r in P.to_numpy().tolist()]
    Pnet = generate_net(population, P_list, covPattern)
    P_net_stat = calc_network_stat(Pnet, Network_stats)
    if print_calculations:
      print("P info:", nx.info(Pnet))
      print("Pnet statistics:", P_net_stat)
  else:
    P_net_stat = 0

  results = [[] for _ in range(samplesize)]
  counter = 0

  for i in range(burnin+samplesize*interval):
    proposal_edge = proposal_edge_func(g)
    g_proposal_edge = g.has_edge(proposal_edge[0], proposal_edge[1])
    g2_proposal_edge = not g_proposal_edge

    g2_net_stat = calc_network_stat_2(Network_stats, proposal_edge, g_net_stat, g2_net_stat, g_proposal_edge, covPattern, g)

    f_g_g2_bool = True
    f_g_g2 = calc_f(Network_stats,g_net_stat, g2_net_stat, proposal_edge, g_proposal_edge, covPattern, bayesian_inference, P_net_stat, g, f_g_g2_bool)

    f_g_g2_bool = False
    f_g2_g = calc_f(Network_stats,g2_net_stat, g_net_stat, proposal_edge, g2_proposal_edge, covPattern, bayesian_inference, P_net_stat, g, f_g_g2_bool)


    if print_calculations:
      print("####Before prob####")
      print("g############")
      print(g_net_stat)
      print(proposal_edge)
      print(g.degree[proposal_edge[0]])
      print(g.degree[proposal_edge[1]])
      print(g_proposal_edge)
      print("g2############")
      print(g2_net_stat)
      print(g2_proposal_edge)
      print("####End prob####")

    prob_g, prob_g2 = calc_probs(g_net_stat, g2_net_stat, proposal_edge, covPattern, Network_stats, Prob_Distr, Prob_Distr_Params, g, g_proposal_edge)

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
      MH_prob = math.inf
    elif math.isnan(prob_g2) or f_g_g2==0:
      MH_prob = -math.inf
    else: 
      MH_prob = math.log(f_g2_g) + prob_g2 - (math.log(f_g_g2) + prob_g)
    
    if print_calculations:
      print(MH_prob)

    if bayesian_inference == 1 and MH_prob < math.inf:
      MH_prob = bayes_inf_MH_prob_calc(MH_prob, g, proposal_edge, Pnet, Ia, Il, R, epi_params)

    if MH_prob >= 0 or math.log(np.random.uniform(0,1,1)) < MH_prob:
      #Accept proposal
      g = proposal_g2(g, proposal_edge)
      g_net_stat = np.copy(g2_net_stat)
    else:   
      #Reject proposal
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
                          G, 
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

  return(Network_stats)


#################################
#################################
#################################

# Network_stats = ["Degree"]
# Prob_Distr = ["Multinomial_Poisson"]
# Prob_Distr_Params = [10, [0.1, 0.6, 0.3, 0.0001]]  

# samplesize = 1000
# burnin = 1000
# interval = 10
# statsonly = True
# G = pd.DataFrame([[1,2],[2,3]])
# P = pd.DataFrame([[1,2],[2,3]])
# population = 10
# covPattern = [0,0,0,0,0,1,1,1,1,1]
# bayesian_inference = False
# Ia = [0,0,0,0,0,1,1,1,1,1] 
# Il = [0,0,0,0,0,1,1,1,1,1]
# R = [0,0,0,0,0,1,1,1,1,1]
# epi_params = [10,5]
# print_calculations = False

# cProfile.run('CCMnet_constr_py(Network_stats, Prob_Distr, Prob_Distr_Params, samplesize, burnin,  interval, statsonly, P, population, covPattern, bayesian_inference,Ia, Il, R,  epi_params, print_calculations)')


# print("Network_stats:", Network_stats, " Type:", type(Network_stats))
# print("Prob_Distr:", Prob_Distr, " Type:", type(Prob_Distr))
# print("Prob_Distr_Params:", Prob_Distr_Params, " Type:", type(Prob_Distr_Params))
# print("samplesize:", samplesize, " Type:", type(samplesize))
# print("burnin:", burnin, " Type:", type(burnin))
# print("interval:", interval, " Type:", type(interval))
# print("statsonly:", statsonly, " Type:", type(statsonly))
# print("G:", G, " Type:", type(G))
# print("G[1]:", G[1], " Type:", type(G[1]))
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
#   Prob_Distr, 
#   Prob_Distr_Params, 
#   samplesize, 
#   burnin,  
#   interval, 
#   statsonly,
#   G, 
#   P, 
#   population, 
#   covPattern, 
#   bayesian_inference,
#   Ia, 
#   Il, 
#   R,  
#   epi_params, 
#   print_calculations)


#print(results)

#rint(type(results))
#print(results[1])
#print(np.mean(results[1], axis=0))

