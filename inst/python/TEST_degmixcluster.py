
import networkx as nx
import pandas as pd
import numpy as np
import random, json
import math
import copy

import sys
import os

target_path = '/Users/ravigoyal/Dropbox/Academic/Research/Projects/CCMnet_package/CCMnet_py/inst/python'

# Add it to the path
if target_path not in sys.path:
  sys.path.append(target_path)


from CCMnet_network_gen import *
from CCMnet_proposal import *
from CCMnet_netprop_degmixcluster import *
from CCMnet_proposal import *

population = 10
covPattern = []
Network_stats = ["degmix_clustering"]
print_calculations = 1
bayesian_inference = 0
P_net_stat = []

# Edges (P) containing multiple triangles
# Triangle 1: nodes 1, 2, 3
# Triangle 2: nodes 4, 5, 6
# Path/Triad: nodes 7, 8, 9 (connecting 7-8 and 8-9, leaving 7-9 open for a proposal)
P = [
    (1, 2), (2, 3), (3, 1),  # Triangle A
    (4, 5), (5, 6), (6, 4),  # Triangle B
    (7, 8), (8, 9),          # Open triad (potential triangle)
    (1, 4), (6, 7), (9, 10)  # Connecting bridges
]

g = generate_net(population, P, covPattern)
g_net_stat = calc_network_stat_degmix_clustering(g)
g2_net_stat = copy.deepcopy(g_net_stat)

print("####Graph information: Begin####")
print(g)
print("####Graph information: End####")
print("####Initial g statistics: Begin####")
print(g_net_stat)
print("####Initial g statistics: End####")

MH_proposal_type = "TNT"

proposal_edge = [7,9]
g_proposal_edge = g.has_edge(proposal_edge[0], proposal_edge[1])
g2_proposal_edge = not g_proposal_edge

g2_net_stat = calc_network_stat_degmix_clustering_2(proposal_edge, g_net_stat, g2_net_stat, g_proposal_edge, covPattern, g, print_calculations)

print("####Proposal information: Begin####")
print("Proposal Edge IDs: ", proposal_edge)
print("Proposal Edge degrees: ", g.degree[proposal_edge[0]], g.degree[proposal_edge[1]])
print("Proposal Edge in g: ", g_proposal_edge)
print("Proposal Edge in g2: ", g2_proposal_edge)
print("####Proposal information: End####")
print("####g2 statistics: Begin####")
print(g2_net_stat)
print("####g2 statistcs: End####")

f_g_g2_bool = True
f_g_g2 = calc_f_degmix_clustering(g_net_stat, proposal_edge, g_proposal_edge, covPattern, bayesian_inference, P_net_stat, g, f_g_g2_bool, print_calculations)

f_g_g2_bool = False
f_g2_g = calc_f_degmix_clustering(g2_net_stat, proposal_edge, g2_proposal_edge, covPattern, bayesian_inference, P_net_stat, g, f_g_g2_bool, print_calculations)

print("####CCM calculations: Begin####")
print("g->g2: ", f_g_g2)
print("g2->g: ", f_g2_g)
print("####CCM calculations: End####")

prob_g, prob_g2 = calc_probs_degmix_clustering(g_net_stat, g2_net_stat, proposal_edge, covPattern, Network_stats, Prob_Distr, Prob_Distr_Params, g, g_proposal_edge, print_calculations)

if print_calculations:
print("####Prob calculations: Begin####")
print("g: ", prob_g)
print("g2: ", prob_g2)
print("####Prob calculations: End####")

############
############
############
############
############
############
###########

 
current_dmm = np.array([
  [40, 74, 95],
  [74,39,78],
  [95,78,23]
  ])
  
  
current_dmm = np.array([
  [40, 75, 95],
  [75,40,75],
  [95,75,23]
  ])

population = 4
current_triangles = 5

k = 1

d_u = 2
d_v = 1

f_g_g2_bool = True

u, v = proposal_edge
current_dmm = g_net_stat['dmm']
current_triangles = g_net_stat['triangles']
  
population = g.number_of_nodes()

prob_f = calc_f_degmix(current_dmm, proposal_edge, g_proposal_edge, covPattern, bayesian_inference, P_net_stat, g, f_g_g2_bool, print_calculations)

if f_g_g2_bool:
  d_u, d_v = g.degree[u], g.degree[v]
else:
  d_u = g.degree[u] + 1 if g_proposal_edge else g.degree[u] - 1
  d_v = g.degree[v] + 1 if g_proposal_edge else g.degree[v] - 1

is_adding = not g_proposal_edge

deg_dist = np.zeros(population)
total_nodes_found = 0
for i in range(population-1):
  row_sum = np.sum(current_dmm[i, :]) + current_dmm[i, i]
  count = int(round(row_sum / (i + 1)))
  deg_dist[i+1] = count
  total_nodes_found += count
deg_dist[0] = max(0, population - total_nodes_found)

k = len(list(nx.common_neighbors(g, u, v)))

if is_adding:
  pa_num = -3.0 * current_triangles
  for deg in range(2, population):
    pa_num += (deg * (deg - 1) / 2.0) * deg_dist[deg]
  
  pa_dem = 0.0
  # i and j represent the degrees (1 to population-1)
  for i in range(1, population):
    for j in range(i, population):
      # 1. Total possible pairs of nodes between degree classes i and j
      if i == j:
        # n_i * (n_i - 1) / 2
        slots = (deg_dist[i] * (deg_dist[i] - 1)) / 2.0
      else:
        # n_i * n_j
        slots = deg_dist[i] * deg_dist[j]
            
      # 2. Number of non-edges (available spots for a new edge)
      # Recall current_dmm index is degree - 1
      non_edges = max(0, slots - current_dmm[i-1, j-1])
            
      # 3. Weight by (i * j)
      # This accounts for how many potential paths of length 2 
      # would be closed by an edge connecting a degree i and degree j node.
      pa_dem += (i * j) * non_edges

  pa = max(1e-15, min(1 - 1e-15, pa_num / pa_dem)) if pa_dem != 0 else 1e-15

  # LOG-STABLE MATH TO PREVENT OVERFLOW

  # 2. Log of Term 1: ln[ (du choose k) * (dv choose k) * k! ]
  # Formula: ln(du!) - ln(k!) - ln(du-k)!) + ln(dv!) - ln(k!) - ln(dv-k)!) + ln(k!)
  # Simplified: ln(du!) + ln(dv!) - ln(k!) - ln(du-k)!) - ln(dv-k)!)
  log_term1 = (gammaln(d_u + 1) + gammaln(d_v + 1) - gammaln(k + 1) - gammaln(d_u - k + 1) - gammaln(d_v - k + 1))

  # 3. Log of Term 2: k * ln(pa)
  log_term2 = k * np.log(pa)

  # 4. Log of Term 3: (du - k) * (dv - k) * ln(1 - pa)
  log_term3 = (d_u - k) * (d_v - k) * np.log(1.0 - pa)

  # Summing for the clustering component of the log probability
  log_p_add = log_term1 + log_term2 + log_term3
    
  # Update total prob_f
  prob_f += log_p_add

else:
  pb_dem = 0.0
  # s and r represent degrees (1 to population-1)
  # current_dmm index is degree - 1
  for r in range(1, population):
    for s in range(1, r+1): # s goes from 1 to r
      # Number of edges between degree r and degree s
      m_rs = current_dmm[r-1, s-1]
            
      # Weighted contribution to the denominator
      pb_dem += (r - 1) * (s - 1) * m_rs
  
  # Equation 21: pb = (3 * T) / pb_dem
  pb = (3.0 * current_triangles) / pb_dem if pb_dem != 0 else 1e-15
  pb = max(1e-15, min(1 - 1e-15, pb))

  # 2. Log of Term 1: ln[ (du-1 choose k) * (dv-1 choose k) * k! ]
  # Formula: ln((du-1)!) + ln((dv-1)!) - ln(k!) - ln((du-1-k)!) - ln((dv-1-k)!)
  log_term1 = (gammaln(d_u) + gammaln(d_v) - gammaln(k + 1) - gammaln(d_u - k) - gammaln(d_v - k))

  # 3. Log of Term 2: k * ln(pb)
  log_term2 = k * np.log(pb)

  # 4. Log of Term 3: (du-1-k) * (dv-1-k) * ln(1 - pb)
  log_term3 = (d_u - 1 - k) * (d_v - 1 - k) * np.log(1.0 - pb)

  # Summing for the clustering component
  log_p_sub = log_term1 + log_term2 + log_term3
    
  # Update total prob_f (which already contains the deg-mix component)
  prob_f += log_p_sub
return prob_f

###################
###################
##################


f_g_g2_bool = True
f_g_g2 = calc_f_degmix_clustering(g_net_stat, proposal_edge, g_proposal_edge, covPattern, bayesian_inference, P_net_stat, g, f_g_g2_bool, print_calculations)

f_g_g2_bool = False
f_g2_g = calc_f_degmix_clustering(g2_net_stat, proposal_edge, g2_proposal_edge, covPattern, bayesian_inference, P_net_stat, g, f_g_g2_bool, print_calculations)

print("####CCM calculations: Begin####")
print("g->g2: ", f_g_g2)
print("g2->g: ", f_g2_g)
print("####CCM calculations: End####")

prob_g, prob_g2 = calc_probs_degmix_clustering(g_net_stat, g2_net_stat, proposal_edge, covPattern, Network_stats, Prob_Distr, Prob_Distr_Params, g, g_proposal_edge, print_calculations)

if print_calculations:
print("####Prob calculations: Begin####")
print("g: ", prob_g)
print("g2: ", prob_g2)
print("####Prob calculations: End####")
