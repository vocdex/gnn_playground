import numpy as np
from scipy.sparse import coo_matrix
from typing import List

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


