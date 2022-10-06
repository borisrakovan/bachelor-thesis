from torch import Tensor
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch import nn

from graph.schemas import InputBatch
from models.base import BaseNet
from models.gnn_decoder import DecoderV1


class GCNBaselineEmb(BaseNet):
    def __init__(
        self,
        num_nodes: int,
        target_size: int,
        embedding_size: int = 100,
        num_message_rounds: int = 3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_nodes, embedding_dim=embedding_size, max_norm=1)

        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)

        self.num_message_rounds = num_message_rounds

        self.decoder = DecoderV1(target_size=target_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, input_batch: InputBatch) -> Tensor:
        geo_batch = input_batch.geo_batch
        x = self.embedding(geo_batch.x).squeeze(1)

        for nr in range(self.num_message_rounds):
            # x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv1(x, geo_batch.edge_index)
            x = F.relu(x)
            # x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, geo_batch.edge_index)

        logits = self.decoder(batch_emb=x, batch=input_batch)
        return logits


class GCNBaseline(BaseNet):
    def __init__(self, embed_dim: int, target_size: int, num_message_rounds: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv1 = GCNConv(embed_dim, embed_dim)
        self.conv2 = GCNConv(embed_dim, embed_dim)

        self.num_message_rounds = num_message_rounds

        self.decoder = DecoderV1(target_size=target_size)

    def forward(self, input_batch: InputBatch) -> Tensor:
        geo_batch = input_batch.geo_batch
        x = geo_batch.x

        for nr in range(self.num_message_rounds):
            # x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv1(x, geo_batch.edge_index)
            x = F.relu(x)
            # x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, geo_batch.edge_index)

        logits = self.decoder(batch_emb=x, batch=input_batch)
        return logits
