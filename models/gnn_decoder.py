import torch
from torch import nn, Tensor

from graph.schemas import InputBatch


class DecoderV1(nn.Module):
    def __init__(self, target_size: int, ):
        super().__init__()
        self.target_size = target_size
        self.linear = None

    def forward(self, batch_emb: Tensor, batch: InputBatch) -> Tensor:
        geo_batch = batch.geo_batch

        embed_dim = batch_emb.size(1)

        batch_size = torch.max(geo_batch.batch).item() + 1
        final_embs = torch.zeros((batch_size, embed_dim * 3))

        for index in range(batch_size):
            graph_emb = batch_emb[geo_batch.batch == index, :]
            node_avg = torch.mean(graph_emb, 0)

            query_edge = batch.targets[index, :]
            final_emb = [graph_emb[query_edge[0]], graph_emb[query_edge[1]], node_avg]
            final_emb = torch.cat(final_emb, dim=0)
            final_embs[index, :] = final_emb

        if self.linear is None:
            self.linear = nn.Linear(final_embs.size(1), self.target_size)

        return self.linear(final_embs)


class DecoderV2(nn.Module):
    def __init__(self, target_size: int, ):
        super().__init__()
        self.target_size = target_size
        self.linear = None

    def forward(self, batch_emb: Tensor, batch: InputBatch) -> Tensor:
        geo_batch = batch.geo_batch

        embed_dim = batch_emb.size(1)

        batch_size = torch.max(geo_batch.batch).item() + 1
        query_embs = torch.zeros((batch_size, embed_dim * 2))

        for index in range(batch_size):
            graph_emb = batch_emb[geo_batch.batch == index, :]
            query_edge = batch.targets[index, :]
            query_emb = [graph_emb[query_edge[0]], graph_emb[query_edge[1]]]
            query_emb = torch.cat(query_emb, dim=0)
            query_embs[index, :] = query_emb

        if self.linear is None:
            self.linear = nn.Linear(query_embs.size(1), self.target_size)

        return self.linear(query_embs)
