from neuralogic.nn import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch
from torch import nn

from graph.schemas import InputBatch
from models.base import BaseNet


class GCN(BaseNet):
    def __init__(self, embed_dim: int, target_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv1 = GCNConv(embed_dim, embed_dim)
        self.conv2 = GCNConv(embed_dim, embed_dim)

        self.target_size = target_size
        self.linear = nn.Linear(embed_dim * 2, self.target_size)

    def forward(self, input_batch: InputBatch) -> Tensor:
        geo_batch = input_batch.geo_batch

        x = self.conv1(geo_batch.x, geo_batch.edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, geo_batch.edge_index)

        batch_size = torch.max(geo_batch.batch).item() + 1
        query_embs = torch.zeros((batch_size, self.embed_dim * 2))

        # todo try embed_dim * 3 with averaged graph emb

        for index in range(batch_size):
            graph_emb = x[geo_batch.batch == index, :]
            query_edge = input_batch.targets[index, :]
            query_emb = [graph_emb[query_edge[0]], graph_emb[query_edge[1]]]
            query_emb = torch.cat(query_emb, dim=0)
            query_embs[index, :] = query_emb

        logits = self.linear(query_embs)

        return logits


class GCN_MP(BaseNet):
    def __init__(self, embed_dim: int, target_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv1 = GCNConv(embed_dim, embed_dim)
        self.conv2 = GCNConv(embed_dim, embed_dim)

        self.target_size = target_size
        self.linear = nn.Linear(embed_dim * 2, self.target_size)

        self.nb_message_rounds = 3

    def forward(self, input_batch: InputBatch) -> Tensor:
        # todo try embed_dim * 3 with averaged graph emb
        geo_batch = input_batch.geo_batch

        for nr in range(self.nb_message_rounds):
            # x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv1(geo_batch.x, geo_batch.edge_index)
            x = F.elu(x)
            # x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, geo_batch.edge_index)

        batch_size = torch.max(geo_batch.batch).item() + 1
        query_embs = torch.zeros((batch_size, self.embed_dim * 2))

        for index in range(batch_size):
            graph_emb = x[geo_batch.batch == index, :]
            query_edge = input_batch.targets[index, :]
            query_emb = [graph_emb[query_edge[0]], graph_emb[query_edge[1]]]
            query_emb = torch.cat(query_emb, dim=0)
            query_embs[index, :] = query_emb

        logits = self.linear(query_embs)

        return logits
