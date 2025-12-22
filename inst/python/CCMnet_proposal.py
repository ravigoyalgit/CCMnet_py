
import networkx as nx
import pandas as pd
import numpy as np
import random, json
import operator as op
from functools import reduce
from itertools import cycle, islice
import math
import cProfile

def proposal_edge_func(g, MH_proposal_type):

  if MH_proposal_type=="random":
    proposal_edge = random.sample(list(g.nodes), 2)
  if MH_proposal_type=="TNT":
    if np.random.uniform(0,1,1) < 0.5 and g.number_of_edges() > 0:
      proposal_edge = list(random.choice(list(g.edges())))     
    else:
      proposal_edge = random.sample(list(g.nodes), 2)

  return proposal_edge

def proposal_g2(g, proposal_edge):

  if g.has_edge(proposal_edge[0], proposal_edge[1]):
    #Remove edge
    g.remove_edge(proposal_edge[0], proposal_edge[1])
  else: 
    #Add edge
    g.add_edge(proposal_edge[0], proposal_edge[1])

  return g
