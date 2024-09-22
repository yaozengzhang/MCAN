import math
import torch as th

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple pygGCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(th.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(th.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, infeatn, adj):
        support = th.spmm(infeatn, self.weight)
        output = th.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN(Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)

        self.gc2 = GraphConvolution(nhid, nclass)

        self.dropout = dropout

    def forward(self, x, adj):
        #print("x", x.shape)
        #print("adj", adj.shape)

        x = self.gc1(x, adj)

        # Specify the folder path and file name
        folder_path = 'F:/原电脑深度学习相关/rumordetection-version.1/原数据文件-twitter和微博数据集(可以操作)/GCN_embeddding'
        file_name = 'GCN_alltweibo_tensor_fold_5.pt'

        #print(x.shape)
        # Save the tensor to file
        th.save(x, f"{folder_path}/{file_name}")

        x = th.relu(x)
        x = th.dropout(x, self.dropout, train=self.training)
        #print(x.shape)
        x = self.gc2(x, adj)
        #print(x.shape)
        return x
