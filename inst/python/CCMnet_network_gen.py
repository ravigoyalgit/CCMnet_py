
import networkx as nx
import pandas as pd
import numpy as np
import random, json
import operator as op
from functools import reduce
from itertools import cycle, islice
import math
import cProfile

def generate_initial_g(population, covPattern):
  
  g = nx.empty_graph(range(1, population + 1))  # 1-based nodes
  nx.set_node_attributes(g, values=dict(zip(range(1, population+1), covPattern)), name='covPattern')

  return g

def generate_net(population, P, covPattern):
  Pnet = nx.Graph()
  Pnet.add_nodes_from(range(1, population + 1))  # ensures all nodes exist
  Pnet.add_edges_from(P)
  nx.set_node_attributes(Pnet, values = dict(zip(list(range(1, population+1)), covPattern)) , name = 'covPattern')

  return Pnet
