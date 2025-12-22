
import networkx as nx
import pandas as pd
import numpy as np
import random, json
import math

from CCMnet_utils import *

#################################
###Calculate initial statistic###
#################################

def calc_network_stat_degmix(g):
  
  n = g.number_of_nodes()
    
  # Get degrees
  deg_dict = dict(g.degree())
    
  # Initialize n x n matrix
  g_net_stat = np.zeros((n, n), dtype=int)
    
  # Fill matrix
  for u, v in g.edges():
    du = deg_dict[u]
    dv = deg_dict[v]
    if du < n and dv < n:  # safeguard
      g_net_stat[du, dv] += 1
      if du != dv:
        g_net_stat[dv, du] += 1  # ensure symmetry

  return  g_net_stat

#################################
########Update statistic#########
#################################

def calc_network_stat_degmix_2(proposal_edge, g_net_stat, g2_net_stat, g_proposal_edge, covPattern, print_calculations):

  deg0 = g.degree[proposal_edge[0]] 
  deg1 = g.degree[proposal_edge[1]]

  if print_calculations:
      print("####In Deg Mix 2####")
      print(deg0)
      print(deg1)
      print(g2_net_stat[deg0][deg1])
      print(g2_net_stat[deg1][deg0])
      print("####In Deg Mix 2####")

  if g_proposal_edge:
    if deg0 == deg1:
      g2_net_stat[deg0][deg1] = g2_net_stat[deg0][deg1] - 1
    else:
      g2_net_stat[deg0][deg1] = g2_net_stat[deg0][deg1] - 1
      g2_net_stat[deg1][deg0] = g2_net_stat[deg1][deg0] - 1
  else:
    if deg0 == deg1:
      g2_net_stat[deg0][deg1] = g2_net_stat[deg0][deg1] + 1
    else:
      g2_net_stat[deg0][deg1] = g2_net_stat[deg0][deg1] + 1
      g2_net_stat[deg1][deg0] = g2_net_stat[deg1][deg0] + 1

  if print_calculations:
      print("####Out Deg Mix 2####")
      print(deg0)
      print(deg1)
      print(g2_net_stat[deg0][deg1])
      print(g2_net_stat[deg1][deg0])
      print("####Out Deg Mix 2####")
      
  return g2_net_stat


##########################################
###Calculate Congruence Class Statistic###
##########################################

def calc_f_degmix(g_net_stat, proposal_edge, g_proposal_edge, covPattern, bayesian_inference, P_net_stat, g, f_g_g2_bool):

  return prob_g_g2  

#####################################
###Calculate statistic probability###
#####################################

def calc_probs_degmix(g_net_stat, g2_net_stat, proposal_edge, covPattern, Prob_Distr, Prob_Distr_Params):

  if (Prob_Distr[0] == "Multinomial_Poisson"):

    x_g = sum(g_net_stat[np.triu_indices(g_net_stat.shape[0])])
    x_g2 = sum(g2_net_stat[np.triu_indices(g2_net_stat.shape[0])])

    deg0 = g.degree[proposal_edge[0]] 
    deg1 = g.degree[proposal_edge[1]]

    entry_id = sum(range(deg1+1)) + deg0

    if x_g > x_g2:
      #removed edge
      log_prob_x_g2 = -math.log(Prob_Distr_Params[0]) + math.log(x_g)
      log_prob_x2_g2 = -math.log(x_g) + math.log(g_net_stat[deg0][deg1]) - math.log(Prob_Distr_Params[1][entry_id])
    else:
      log_prob_x_g2 = math.log(Prob_Distr_Params[0]) - math.log(x_g2)
      log_prob_x2_g2 = math.log(x_g2) - math.log(g2_net_stat[deg0][deg1]) + math.log(Prob_Distr_Params[1][entry_id])

    prob_g = 0
    prob_g2 = log_prob_x_g2 + log_prob_x2_g2

  return prob_g, prob_g2

