import numpy as np
from scipy.sparse import coo_matrix
from typing import List
import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt


def edges_to_adjacency_mat(edges:List, num_nodes:int)->np.array:
    """Build adjacency matrix from edge lists"""
    row = [edge[0] for edge in edges]
    col = [edge[1] for edge in edges]
    data = np.ones(len(edges))

    adj_mat = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    return adj_mat.toarray()

def adjacency_mat_to_edges(adj_mat:np.array)->List:
    """Build edge lists from adjacency matrix"""
    edges = []
    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            if adj_mat[i, j] == 1:
                edges.append([i, j])
    return edges

def load_cora(path:str):
    """Load the Cora dataset"""
    edgelist = pd.read_csv(os.path.join(path, 'cora.cites'), sep='\t', header=None, names=['target', 'source'])
    edgelist["label"] = "cites"
    features = pd.read_csv(os.path.join(path, 'cora.content'), sep='\t', header=None)
    features = features.set_index(0)
    labels = features[features.columns[-1]]
    features = features.drop(features.columns[-1], axis=1)
    features = features.values
    labels = labels.values
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
