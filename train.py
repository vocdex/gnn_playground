# Train Graph Convolutional Network on the Cora dataset
# Usage: python train.py
from model import GCN
from utils import load_cora, accuracy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Load the Cora dataset
adj_matrix, features, labels, train_mask, val_mask, test_mask = load_cora()

# Hyperparameters
in_features = features.shape[1]
hidden_features = 16
out_features = labels.max().item() + 1
num_epochs = 100
lr = 0.0001

def train():
    # Initialize the model
    model = GCN(in_features, hidden_features, out_features)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj_matrix)
        loss = criterion(output[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            train_acc = accuracy(output[train_mask], labels[train_mask])
            val_acc = accuracy(output[val_mask], labels[val_mask])
            test_acc = accuracy(output[test_mask], labels[test_mask])
            print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}, Train accuracy: {train_acc:.4f}, Val accuracy: {val_acc:.4f}, Test accuracy: {test_acc:.4f}')
        
if __name__ == '__main__':
    train()
