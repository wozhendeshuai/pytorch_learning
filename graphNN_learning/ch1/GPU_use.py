import dgl
import torch as th

print(th.cuda.is_available())
u, v = th.tensor([0, 1, 2]), th.tensor([2, 3, 4])
g = dgl.graph((u, v))
g.ndata['x'] = th.randn(5, 3)
print(g.device)
cuda_g = g.to('cuda:0')
print(cuda_g.device)
print(cuda_g.ndata['x'].device)

u,v=u.to('cuda:0'),v.to('cuda:0')
g=dgl.graph((u,v))
print(g.device)

print(cuda_g.in_degrees())
print(cuda_g.in_edges([2, 3, 4]))
print(cuda_g.in_edges(th.tensor([2, 3, 4]).to('cuda:0')))
# cuda_g.ndata['h']=th.randn(5,4)
