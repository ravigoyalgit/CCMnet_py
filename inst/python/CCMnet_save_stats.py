
import networkx as nx
import pandas as pd
import numpy as np
import random, json
import operator as op
from functools import reduce
from itertools import cycle, islice
import math
import cProfile

# def save_stats(g_net_stat, results, counter, Network_stats, g):
#   if Network_stats[0].strip().lower() == "edge" and len(Network_stats) == 1:
#     g_net_stat_temp = nx.degree_histogram(g)
#     g_net_stat_temp.extend([0] * (g.number_of_nodes() - len(g_net_stat_temp)))
#     g_net_stat_temp.append(g_net_stat)
#     results[counter] =  g_net_stat_temp
#   if Network_stats[0].strip().lower() == "mixing" and len(Network_stats) == 1:
#     results[counter] = g_net_stat[np.triu_indices(g_net_stat.shape[0])]
#   if Network_stats[0].strip().lower() == "degree" and len(Network_stats) == 1:
#     results[counter] = g_net_stat

def save_stats(g_net_stat, results, counter, Network_stats, g, Obs_stats):

  stat = Network_stats[0].strip().lower()
    
  if Obs_stats is None:
    obs = "none"
  else:
    obs  = Obs_stats[0].strip().lower()

  row = []

  # ----------------------
  # Network statistic
  # ----------------------
  if stat == "edge":
    row.append(g_net_stat)

  elif stat == "mixing":
    row.extend(
      g_net_stat[np.triu_indices(g_net_stat.shape[0])].tolist()
    )

  elif stat == "degree":
    row.extend(g_net_stat.tolist())

  elif stat == "degmix":
    row.extend(
      g_net_stat[np.triu_indices(g_net_stat.shape[0])].tolist()
    )

  elif stat == "degmix_clustering":
    g_net_stat_dmm = g_net_stat['dmm']
    g_net_stat_tri = g_net_stat['triangles']
    row.extend(
      g_net_stat_dmm[np.triu_indices(g_net_stat_dmm.shape[0])].tolist()
    )
    # 2. Append the triangle scalar at the very end
    row.append(g_net_stat_tri)

  else:
    raise NotImplementedError(f"Network_stats={Network_stats}")

  # ----------------------
  # Observed statistic
  # ----------------------
  if obs == "degree":
    deg_hist = nx.degree_histogram(g)
    deg_hist.extend([0] * (g.number_of_nodes() - len(deg_hist)))
    row.extend(deg_hist)
  elif obs == "none":
    # No observed statistics requested
    pass
  else:
    raise NotImplementedError(f"Obs_stats={Obs_stats}")

  results[counter] = row
