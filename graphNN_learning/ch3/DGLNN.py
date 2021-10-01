import torch.nn as nn
from dgl.utils import  expand_as_pair
class SAGEConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SAGEConv,self).__init__()

        self._in_src_feats,self._in_dst_feats=expand_as_pair(in_feats)
        self._out_feats=out_feats
        self._aggre_type=aggregator_type
        self.norm=norm
        self.activation=activation
        # 聚合类型：mean，max_pool,lstm,gcn
        if aggregator_type not in ['mean', 'max_pool', 'lstm', 'gcn']:
            raise KeyError('Aggregator type{} not supported.'.format(aggregator_type))
        if aggregator_type=='max_pool':
            self.fc_pool=nn.linear(self._in_src_feats,self._in_src_feats)
        if aggregator_type=='lstm':
            self.lstm=nn.LSTM(self._in_src_feats,self._in_src_feats,batch_first=True)
        if aggregator_type in ['mean', 'max_pool', 'lstm']:
            self.fc_self=nn.Linear(self._in_dst_feats,out_feats,bias=bias)
        self.fc_neigh=nn.Linear(self._in_src_feats,out_feats,bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        gain=nn.init.calculate_gain('relu')
        if self._aggre_type=='max_pool':
            nn.init.xavier_uniform_(self.fc_pool.weight,gain=gain)
        if self._aggre_type=='lstm':
            self.lstm.reset_parameters()
        if self._aggre_type!='gcn':
            nn.init.xavier_uniform_(self.fc_self.weight,gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight,gain=gain)

    def forward(self,graph,feat):
        with graph.local_scope():
            feat_src,feat_dst=expand_as_pair(feat,graph)

    def expand_as_pair(input_,g=None):
        if isinstance(input_,tuple):
            return input_
        elif g is not None and g.is_block:
            if isinstance(input_,Mapping):
                input_dst={
                    k: F.narrow_row(v,0,g.number_of_dst_nodes(k))
                    for k,v in input_.items()
                }
            else:
                input_dst=F.narrow_row(input_,0,g.number_of_dst_nodes())
            return input_,input_dst
        else:
            return input_,input_

import dgl.function as fn
import torch.nn.functional as F
from dgl.utils import check_eq_shape

if self._aggre_type=='mean':
    graph.srcdata['h']=feat_src
    graph.update_all(fn.copy_u('h','m'),fn.mean('m','neigh'))
    h_neigh=graph.dstdata['neigh']
elif self._aggre_type=='gcn':
    check_eq_shape(feat)
    graph.srcdata['h']=feat_src
    graph.dstdata['h']=feat_dst
    graph.update_all(fn.copy_u('h','m'),fn.sum('m','neigh'))
    degs=graph.in_degrees().to(feat_dst)
    h_neigh=(graph.dstdata['neigh']+graph.dstdata['h'])/(degs.unsqueeze(-1) + 1)
else:
    raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

if self._aggre_type=='gcn':
    rst=self.fc_neigh(h_neigh)
else:
    rst=self.fc_self(h_self)+self.fc_neigh(h_neigh)



