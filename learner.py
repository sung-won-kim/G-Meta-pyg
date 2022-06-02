import torch
import torch.nn.functional as F
import dgl.function as fn
import torch.nn as nn
from torch.nn import init
from torch_geometric.utils import degree
from torch_geometric.nn import MessagePassing

# Sends a message of node feature h.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
msg = fn.copy_src(src='h', out='m')


class msg(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).

    def forward(self, x, edge_index):

        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j


msg = msg()


class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation=None):
        super(GraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = True
        self._activation = activation

    def forward(self, n_subgraph, e_subgraph, feat, weight, bias):

        if self._norm:
            degrees = degree(e_subgraph[0], n_subgraph.shape[0])
            norm = torch.pow(degrees.float().clamp(min=1), -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp).to(feat.device)
            feat = feat * norm

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            feat = torch.matmul(feat, weight)
            rst = msg(feat, e_subgraph)
        else:
            # aggregate first then mult W
            rst = msg(feat, e_subgraph)
            rst = torch.matmul(rst, weight)

        rst = rst * norm

        rst = rst + bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)


class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()

        self.vars = nn.ParameterList()
        self.graph_conv = []
        self.config = config
        self.LinkPred_mode = False

        for i, (name, param) in enumerate(self.config):

            if name is 'Linear':
                if self.LinkPred_mode:
                    w = nn.Parameter(torch.ones(param[1], param[0] * 2))
                else:
                    w = nn.Parameter(torch.ones(param[1], param[0]))
                init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
            if name is 'GraphConv':
                # param: in_dim, hidden_dim
                w = nn.Parameter(torch.Tensor(param[0], param[1]))
                init.xavier_uniform_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
                self.graph_conv.append(
                    GraphConv(param[0], param[1], activation=F.relu))

    def forward(self, n_subgraph, e_subgraph, to_fetch, features, vars=None):
        # For undirected graphs, in_degree is the same as
        # out_degree.

        if vars is None:
            vars = self.vars

        idx = 0
        idx_gcn = 0

        h = features.float()
        # h = h.to(device)

        for name, param in self.config:
            if name is 'GraphConv':
                w, b = vars[idx], vars[idx + 1]
                conv = self.graph_conv[idx_gcn]

                h = conv(n_subgraph, e_subgraph, h, w, b)

                idx += 2
                idx_gcn += 1

            if name is 'Linear':
                w, b = vars[idx], vars[idx + 1]
                h = F.linear(h, w, b)
                idx += 2

        return h, h

    def zero_grad(self, vars=None):

        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars
