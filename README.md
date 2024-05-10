# gnn_playground
An ever-growing playground of scripts to showcase GNN's power and might
To install the required packages, run the following command:
```bash
pip install torch torchvision torch-geometric 
pip install matplotlib 
pip install natsort
```

## Karate Club Dataset
It seems like the ReLU activation function is not optimal for GCNs as the model does not learn proper embeddings. On the other hand, tahnh activation function seems to work well. The model is able to learn proper embeddings and the communities are clearly visible in the 2D plot.
To run the code, simply run the following command:
```bash
python pyg_intor.py
```
Here is the plot for the Karate Club dataset using the GCN model with tanh activation function:
![Karate Club Dataset](/results/tanh/animation_tanh.gif)

On the other hand, here is the plot for the Karate Club dataset using the GCN model with ReLU activation function:
![Karate Club Dataset](/results/relu/animation_relu.gif)

## Goals
- Implement a simple GCN, GAT models from scratch in PyTorch
- Train these models on a variety of datasets from [SNAP](https://snap.stanford.edu/data/index.html) and document the results.
- Visualize the learned embeddings in real-time using plotly on a 2D plane. 
- If 2D visualization is not possible, use t-SNE to reduce the dimensionality of the embeddings and visualize them in 2D.

## Ambitious Goals
- Implement GNN dynamics model for common Mujoco robotics environments.
- Do ablation studies to compare common MLP dynamics models with GNN dynamics models.
- Visualize these embeddings in real-time using plotly on a 2D plane too, if possible.
