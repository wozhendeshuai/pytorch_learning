def message_func(edges):
    return {'he':edges.src['hu']+edges.dst['hv']}
import torch
def reduce_func(nodes):
    return {'h':torch.sum(nodes.mailbox['m'],dim=1)}
import dgl.function as fn
import dgl
import torch as th

# 边 0->1, 0->2, 0->3, 1->3
u,v=th.tensor([0,0,0,1]),th.tensor([1,2,3,3])
g=dgl.graph((u,v))
g.apply_edges(fn.u_add_v('el','er','e'))

def updata_all_example(graph):
    graph.update_all(fn.u_mul_e('ft','a','m'),
                     fn.sum('m','ft'))
    final_ft=graph.ndata['ft']*2
    return final_ft
node_feat_dim=th.tensor([123,123,123])
out_dim=th.tensor([1])
linear=th.nn.Parameter(torch.FloatTensor(size=(node_feat_dim*2,out_dim)))
def concat_message_function(edges):
    return {'cat_feat':torch.cat([edges.src.ndata['feat'],edges.dst.ndata['feat']])}
g.apply_edges(concat_message_function)
g.edata['out']=g.edata['cat_feat']
import dgl.function as fn

linear_src = th.nn.Parameter(torch.FloatTensor(size=(node_feat_dim, out_dim)))
linear_dst = th.nn.Parameter(torch.FloatTensor(size=(node_feat_dim, out_dim)))
out_src = g.ndata['feat'] @ linear_src
out_dst = g.ndata['feat'] @ linear_dst
g.srcdata.update({'out_src': out_src})
g.dstdata.update({'out_dst': out_dst})
g.apply_edges(fn.u_add_v('out_src', 'out_dst', 'out'))