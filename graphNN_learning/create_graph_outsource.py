import dgl
import torch as th
import scipy.sparse as sp
spmat=sp.rand(100,100,density=0.05)
print(spmat)
print(dgl.from_scipy(spmat))

import networkx as nx
nx_g=nx.path_graph(5)
print(nx_g)
print(dgl.from_networkx(nx_g))