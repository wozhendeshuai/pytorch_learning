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


