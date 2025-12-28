
import networkx as nx
import pandas as pd
import numpy as np
import random, json
import math

from CCMnet_utils import *

#################################
###Calculate initial statistic###
#################################

def calc_network_stat_degmixcluster(g):
  
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

  return g_net_stat
#################################
########Update statistic#########
#################################



##########################################
###Calculate Congruence Class Statistic###
##########################################


#####################################
###Calculate statistic probability###
#####################################

