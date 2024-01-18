import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.module):
    """Basic GCN layer from Kipf et al. (2017)"""
    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    
    def forward(self, node_features, adj_matrix):
        # node_features: (batch_size, num_nodes, in_features)
        # adj_matrix: (batch_size, num_nodes, num_nodes)
        # output: (batch_size, num_nodes, out_features)
        num_neighbours = adj_matrix.sum(dim=-1, keepdim=True)
        node_features = self.linear(node_features)
        node_features = torch.bmm(adj_matrix, node_features)
        node_features = node_features / num_neighbours
        return node_features
    
class GCN(nn.Module):
    """GCN from Kipf et al. (2017)"""
    def __init__(self, in_features, hidden_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.layer1 = GCNLayer(in_features, hidden_features, bias=bias)
        self.layer2 = GCNLayer(hidden_features, out_features, bias=bias)
    
    def forward(self, node_features, adj_matrix):
        # node_features: (batch_size, num_nodes, in_features)
        # adj_matrix: (batch_size, num_nodes, num_nodes)
        # output: (batch_size, num_nodes, out_features)
        node_features = F.relu(self.layer1(node_features, adj_matrix))
        node_features = self.layer2(node_features, adj_matrix)
        return node_features
    
