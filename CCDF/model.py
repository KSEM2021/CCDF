import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter

class GraphConvolution(Module):
    """
    A Graph Convolution Layer (GCN)
    """
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input_feature, adj):
        support = torch.mm(input_feature, self.weight)
        output = torch.spmm(adj, support)
        return output


class CCDF(nn.Module):
    def __init__(self, nfeat, nhid, nclass, k,dropout):
        super(CCDF, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.gc3 = GraphConvolution(nclass, nclass)
        self.lin = nn.Linear(nfeat,nclass)
        self.dropout = dropout
        self.k = k

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x1 = F.relu(self.gc1(x, adj))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.softmax(self.gc2(x1,adj))
        h0 = F.softmax(self.lin(x))
        x = torch.add(x2,self.gc3(h0,self.k))
        return x

def get_model(model_opt, nfeat, nclass, nhid,k, dropout,cuda=True):
    if model_opt == "CCDF":
        model = CCDF(nfeat=nfeat,
                    nhid=nhid,
                    nclass=nclass,
                    dropout=dropout,
                    k=k
                    )
    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))

    if cuda: model.cuda()
    return model
