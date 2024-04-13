import numpy as np
from scipy.sparse import coo_matrix
from typing import List
import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt


def edges_to_adjacency_mat(edges, num_nodes):
    """Convert edge list to adjacency matrix"""
    adj_mat = np.zeros((num_nodes, num_nodes))
    for edge in edges:
        adj_mat[edge[0]-1, edge[1]-1] = 1
        adj_mat[edge[1]-1, edge[0]-1] = 1  # Assuming the graph is undirected
    return adj_mat


def load_cora(path:str):
    """Load the Cora dataset"""
    edgelist = pd.read_csv(os.path.join(path, 'cora.cites'), sep='\t', header=None, names=['target', 'source'])
    features = pd.read_csv(os.path.join(path, 'cora.content'), sep='\t', header=None)
    labels = features[features.columns[-1]]
    breakpoint()
    return edgelist, features, labels


def visualize_cora(path:str):
    """Visualize the Cora dataset"""
    edgelist, features, labels = load_cora(path)
    adj_mat = edges_to_adjacency_mat(edgelist[['source', 'target']].values, features.shape[0])
    G = nx.from_numpy_matrix(adj_mat)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, node_size=50, node_color=labels, cmap=plt.cm.tab10)
    plt.show()
