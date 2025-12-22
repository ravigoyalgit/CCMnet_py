
import networkx as nx
import pandas as pd
import numpy as np
import random, json
import math

from CCMnet_utils import *

#################################
###Calculate initial statistic###
#################################

def calc_network_stat_mixing(g):

  n_cov = len(set(nx.get_node_attributes(g, "covPattern").values()))
  g_net_stat = np.asarray(nx.attr_matrix(g, node_attr="covPattern", normalized=False, rc_order=range(n_cov)))
  return g_net_stat

#################################
########Update statistic#########
#################################

def calc_network_stat_mixing_2(proposal_edge, g_net_stat, g2_net_stat, g_proposal_edge, covPattern, print_calculations):

  cov0 = covPattern[proposal_edge[0]-1]
  cov1 = covPattern[proposal_edge[1]-1]

  if print_calculations:
      print("####In Mixing 2####")
      print(cov0)
      print(cov1)
      print(g2_net_stat[cov0][cov1])
      print(g2_net_stat[cov1][cov0])
      print("####In Mixing 2####")

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

  if print_calculations:
      print("####Out Mixing 2####")
      print(cov0)
      print(cov1)
      print(g2_net_stat[cov0][cov1])
      print(g2_net_stat[cov1][cov0])
      print("####Out Mixing 2####")
      
  return g2_net_stat


##########################################
###Calculate Congruence Class Statistic###
##########################################

def calc_f_mixing(g_net_stat, proposal_edge, g_proposal_edge, covPattern, bayesian_inference, P_net_stat):

  cov0 = covPattern[proposal_edge[0]-1]
  cov1 = covPattern[proposal_edge[1]-1]

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


#####################################
###Calculate statistic probability###
#####################################

def calc_probs_mixing(g_net_stat, g2_net_stat, proposal_edge, covPattern, Prob_Distr, Prob_Distr_Params):

  if (Prob_Distr[0] == "NP"):
    x_g = sum(g_net_stat[np.triu_indices(g_net_stat.shape[0])])
    x_g2 = sum(g2_net_stat[np.triu_indices(g2_net_stat.shape[0])])

    cov0 = covPattern[proposal_edge[0]-1]
    cov1 = covPattern[proposal_edge[1]-1]
    
    g_val = int(round(g_net_stat[cov0][cov1]))
    entry_id = sum(range(cov1+1)) + cov0 

#    print("x_g:", x_g)
#    print("x_g2:", x_g2)
#    print("cov0:", cov0)
#    print("cov1:", cov1)
#    print("g_val", g_val)
#    print("entry_id", entry_id)

    if x_g > x_g2:
      #removed edge
      prob_g2 = math.log(Prob_Distr_Params[0][g_val-1][entry_id]) - math.log(Prob_Distr_Params[0][g_val][entry_id])
    else:
      prob_g2 = math.log(Prob_Distr_Params[0][g_val+1][entry_id]) - math.log(Prob_Distr_Params[0][g_val][entry_id])
    prob_g = 0

  else:
    x_g = sum(g_net_stat[np.triu_indices(g_net_stat.shape[0])])
    x_g2 = sum(g2_net_stat[np.triu_indices(g2_net_stat.shape[0])])

    cov0 = covPattern[proposal_edge[0]-1]
    cov1 = covPattern[proposal_edge[1]-1]

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


