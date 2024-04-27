import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch.nn import Linear
from torch_geometric.nn import GCNConv

def visualize_graph(G, color):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2",node_size=70)
    plt.show()
    plt.close()

def visualize_embedding(h, color, epoch=None, loss=None, activation=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    save_dir = f'./results/{activation}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f'{save_dir}/{epoch}.png')
    plt.close()

def make_gif(activation):
    import natsort
    import imageio.v2 as imageio
    import glob
    images = []
    image_dir = f'./results/{activation}'
    filenames = glob.glob(f'{image_dir}/*.png')
    filenames = natsort.natsorted(filenames)
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(f'{image_dir}/animation_{activation}.gif', images, fps=10)
    print(f"Animation saved at: {image_dir}/animation_{activation}.gif")


def load_karate_club(verbose=False):
    from torch_geometric.datasets import KarateClub
    dataset = KarateClub()
    data = dataset[0]

    print(f'Dataset: {dataset}:')  
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    if verbose:
        print(data)
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"Average node degree: {data.num_edges / data.num_nodes:.3f}")
        print(f"Number of training nodes: {data.train_mask.sum()}")
        print(f"Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}")
        print(f"Contains isolated nodes: {data.has_isolated_nodes()}")
        print(f"Contains self-loops: {data.has_self_loops()}")
        print(f"Is undirected: {data.is_undirected()}")
    return dataset

class GCN(torch.nn.Module):
    def __init__(self, dataset, activation: str = 'tanh'):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)
        if activation == 'tanh':
            self.activation = torch.nn.Tanh()
        elif activation == 'relu':
            self.activation = torch.nn.ReLU()
    
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.activation(h)
        h = self.conv2(h, edge_index)
        h = self.activation(h)
        h = self.conv3(h, edge_index)
        h = self.activation(h)
        out = self.classifier(h)
        return out, h

def train_gcn(model, data, activation):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(401):
        optimizer.zero_grad()
        out, h = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        if epoch % 10 == 0:
            visualize_embedding(h, color=data.y, epoch=epoch, loss=loss, activation=activation)
        loss.backward()
        optimizer.step()

def main():
    dataset = load_karate_club(verbose=True)
    data = dataset[0]
    from torch_geometric.utils import to_networkx
    G = to_networkx(data, to_undirected=True)
    visualize_graph(G, color=data.y)
    activation = 'tanh'
    model = GCN(dataset, activation=activation)
    train_gcn(model, data, activation=activation)
    make_gif(activation=activation)

if __name__ == '__main__':
    main()
