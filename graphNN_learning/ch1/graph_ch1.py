import dgl
import torch as th

# 边 0->1, 0->2, 0->3, 1->3
u,v=th.tensor([0,0,0,1]),th.tensor([1,2,3,3])
g=dgl.graph((u,v))
print(g)
# 获取节点的ID
print(g.nodes())
# 获取边的对应端点
print(g.edges())
# 获取边的对应端点和边ID
print(g.edges(form='all'))
#如果具有最大ID的节点没有边，在创建图的时候，用户需要明确地指明节点的数量。
g=dgl.graph((u,v),num_nodes=8)

bg =dgl.to_bidirected(g)
print(bg.edges())

###=====================
g=dgl.graph(([0,0,1,5],[1,2,2,0]))
print(g)
#长度为3的节点特征
g.ndata['x']=th.ones(g.num_nodes(),3)
g.edata['x']=th.ones(g.num_edges(),dtype=th.int32)
print(g)
g.ndata['y']=th.randn(g.num_nodes(),5)
print(g.ndata['x'][1])
print(g.edata['x'][th.tensor([0, 3])])

edges=th.tensor([0,0,0,1]),th.tensor([1,2,3,3])
weights=th.tensor([0.1,0.6,0.9,0.7])
g=dgl.graph(edges)
g.edata['w']=weights
print(g)
