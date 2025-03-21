import torch
from torch_geometric.nn import GATConv

class MultiLayerGATFeatureAggregator(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=4, dropout=0.1, activation="leaky_relu", num_layers=2):

        super(MultiLayerGATFeatureAggregator, self).__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()

        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = torch.nn.LeakyReLU(negative_slope=0.2)
        elif activation == "elu":
            self.activation = torch.nn.ELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.layers.append(
            GATConv(in_dim, hidden_dim, heads=num_heads, concat=True, dropout=dropout)
        )

        for _ in range(num_layers - 2):
            self.layers.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True, dropout=dropout)
            )

        self.layers.append(
            GATConv(hidden_dim * num_heads, out_dim, heads=1, concat=False, dropout=dropout)
        )

        self.dropout = torch.nn.Dropout(dropout)
        self.residual_connection = (
            torch.nn.Linear(in_dim, out_dim) if in_dim != out_dim else None
        )

    def forward(self, features, edge_index):
        x = features
        for layer in self.layers:
            x = layer(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)

        if self.residual_connection is not None:
            residual = self.residual_connection(features)
        else:
            residual = features

        x = x + residual
        return x


def preprocess_edge_index(edge_index, num_nodes):

    edge_index[0] = torch.clamp(edge_index[0], min=0, max=num_nodes - 1)
    edge_index[1] = torch.clamp(edge_index[1], min=0, max=num_nodes - 1)

    valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
    return edge_index[:, valid_mask]
