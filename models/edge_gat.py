"""
Disclaimer: this architecture was inspired by the code for this paper:
https://arxiv.org/pdf/2007.06477.pdf
"""

import torch
from torch import nn, Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros

from graph.schemas import InputBatch
from models.decoder import DecoderV1


class EdgeGATConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        heads: int = 3,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        # change: node_dim = 0
        super(EdgeGATConv, self).__init__(aggr='add', node_dim=0)
        # super(EdgeGATConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels + edge_dim))
        self.edge_update_w = Parameter(torch.Tensor(out_channels + edge_dim, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        glorot(self.edge_update_w)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # [i for i in range(edge_index.size(1)) if edge_index[0, i] == edge_index[1, i]]

        edge_index_1, _ = remove_self_loops(edge_index)
        edge_index_2, _ = add_self_loops(edge_index_1, num_nodes=x.size(0))
        self_loop_edges = torch.zeros(x.size(0), edge_attr.size(1)).to(edge_index_2.device)
        edge_attr_2 = torch.cat([edge_attr, self_loop_edges], dim=0)
        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        return self.propagate(edge_index_2, size=None, x=x, edge_attr=edge_attr_2)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        edge_attr = edge_attr.unsqueeze(1).repeat(1, self.heads, 1)
        x_j = torch.cat([x_j, edge_attr], dim=-1)
        # Expected size 2205 but got size 2206 for tensor number 1 in the list
        x_i = x_i.view(-1, self.heads, self.out_channels)
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = torch.mm(aggr_out, self.edge_update_w)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


class GATEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_edge_types: int,
        embedding_dim: int = 100,
        edge_dim: int = 20,
        num_heads: int = 3,
        dropout: float = 0.0,
        num_message_rounds: int = 3
    ):
        super().__init__()
        self.num_message_rounds = num_message_rounds
        self.embedding = torch.nn.Embedding(num_embeddings=num_nodes, embedding_dim=embedding_dim, max_norm=1)
        self.edge_embedding = torch.nn.Embedding(num_embeddings=num_edge_types, embedding_dim=edge_dim)
        self.att1 = EdgeGATConv(embedding_dim, embedding_dim, edge_dim, heads=num_heads, dropout=dropout)
        self.att2 = EdgeGATConv(embedding_dim, embedding_dim, edge_dim)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight)

    def forward(self, batch) -> Tensor:
        x = self.embedding(batch.x).squeeze(1)
        edge_attr = self.edge_embedding(batch.edge_attr).squeeze(1)
        for nr in range(self.num_message_rounds):
            x = F.dropout(x, p=0.6, training=self.training)
            x = F.elu(self.att1(x, batch.edge_index, edge_attr))
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.att2(x, batch.edge_index, edge_attr)
        return x


class EdgeGAT(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_edge_types: int,
        target_size: int,
        num_heads: int = 3,
        embedding_size: int = 100,
        edge_embedding_size: int = 20,
        num_rounds: int = 3
    ):
        super().__init__()
        self.encoder = GATEncoder(num_nodes=num_nodes, num_edge_types=num_edge_types, num_heads=num_heads,
                                  embedding_dim=embedding_size, edge_dim=edge_embedding_size,
                                  num_message_rounds=num_rounds)
        self.decoder = DecoderV1(target_size=target_size)

    def forward(self, batch: InputBatch) -> Tensor:
        batch_emb = self.encoder(batch.geo_batch)
        logits = self.decoder(batch_emb, batch)
        return logits

