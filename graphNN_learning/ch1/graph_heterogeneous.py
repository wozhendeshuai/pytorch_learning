import dgl
import torch as th
graph_data={
    ('drug','interacts','drug'):(th.tensor([0,1]),th.tensor([1,2])),
    ('drug','interacts','gene'):(th.tensor([0,1]),th.tensor([2,3])),
    ('drug','treats','disease'):(th.tensor([1]),th.tensor([2])),
}
g=dgl.heterograph(graph_data)
print(g.ntypes)
print(g.etypes)
print(g.canonical_etypes)

print(g.metagraph().edges())
g.nodes['drug'].data['hv']=th.ones(3,1)
print(g.nodes['drug'].data['hv'])

g.edges['treats'].data['he']=th.zeros(1,1)
print(g.edges['treats'].data['he'])
import networkx as mx
import  matplotlib.pyplot as plt
mx.draw(g.to_networkx(),with_labels=True)
plt.show()