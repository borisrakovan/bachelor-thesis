"""
Disclaimer: this architecture was inspired by the code for this repo:
https://github.com/uclnlp/ctp
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add

from graph.schemas import InputBatch
from models.base import BaseNet
from models.gnn_decoder import DecoderV1


class EdgeGCNConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        **kwargs
    ):
        super().__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.edge_update_w = Parameter(torch.Tensor(out_channels + edge_dim, out_channels))

        self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.edge_update_w)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr, edge_weight=None):
        x = torch.matmul(x, self.weight)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=x.dtype, device=edge_index.device)

        fill_value = 1
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, x.size(0))

        self_loop_edges = torch.zeros(x.size(0), edge_attr.size(1)).to(edge_index.device)
        edge_attr = torch.cat([edge_attr, self_loop_edges], dim=0)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)

    def message(self, x_j, edge_attr, norm):
        x_j = torch.cat([x_j, edge_attr], dim=-1)
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        aggr_out = torch.mm(aggr_out, self.edge_update_w) + self.bias
        return aggr_out


class EdgeGCN(BaseNet):
    def __init__(
        self,
        num_nodes: int,
        num_edge_types: int,
        target_size: int,
        embedding_size: int = 100,
        edge_embedding_size: int = 20,
        num_rounds: int = 3
    ):
        super().__init__()
        self.num_message_rounds = num_rounds
        self.embedding = nn.Embedding(num_embeddings=num_nodes, embedding_dim=embedding_size, max_norm=1)
        self.edge_embedding = nn.Embedding(num_edge_types, edge_embedding_size)
        self.conv1 = EdgeGCNConv(embedding_size, embedding_size, edge_embedding_size)
        self.conv2 = EdgeGCNConv(embedding_size, embedding_size, edge_embedding_size)

        self.decoder = DecoderV1(target_size=target_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.edge_embedding.weight)

    def forward(self, batch: InputBatch) -> Tensor:
        geo_batch = batch.geo_batch
        x = self.embedding(geo_batch.x).squeeze(1)
        edge_attr = self.edge_embedding(geo_batch.edge_attr).squeeze(1)
        for nr in range(self.num_message_rounds):
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv1(x, geo_batch.edge_index, edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, geo_batch.edge_index, edge_attr)
        logits = self.decoder(batch_emb=x, batch=batch)
        return logits
