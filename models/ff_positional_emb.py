import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from graph.schemas import InputBatch
from models.base import BaseNet

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
from utils import pad_sequences


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class FeedForwardPos(BaseNet):
    UNK_TOKEN = "UNK"

    def __init__(
        self,
        target_size: int,
        relation_lst: list[str],
        hidden_size: int = 80,
        emb_dim: int = 80,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.relation_lst = relation_lst
        self.lin1 = nn.Linear(in_features=self.emb_dim, out_features=hidden_size)
        self.lin2 = nn.Linear(in_features=hidden_size, out_features=target_size)
        self.token_to_idx = {rel: idx for idx, rel in enumerate(self.relation_lst + [self.UNK_TOKEN])}
        self.num_tokens = len(self.token_to_idx)
        self.pos_emb = PositionalEncoding(self.emb_dim)
        self.embedding = nn.Embedding(self.num_tokens, self.emb_dim)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_batch: InputBatch) -> Tensor:
        device = input_batch.targets.device

        batch_tokens = []
        for idx, instance in enumerate(input_batch.instances):
            # encode only the relations
            seq = [t[1] for t in instance.story]
            tokens = [self.token_to_idx[s] for s in seq]

            # counts = Counter(tokens)
            # bag_of_relations = [
            #     counts[i] for i in range(self.num_tokens)
            # ]
            # one_hot_relations = [
            #     1 if counts[i] > 0 else 0 for i in range(self.num_tokens)
            # ]
            batch_tokens.append(tokens)

        batch_token_padded = pad_sequences(batch_tokens, value=self.token_to_idx[self.UNK_TOKEN])
        batch_token_tensor = torch.tensor(batch_token_padded, dtype=torch.long, device=device)

        batch_emb = self.embedding(batch_token_tensor) * math.sqrt(self.emb_dim)

        x = batch_emb
        x = x.transpose(0, 1)
        x = self.pos_emb(x)
        x = x.mean(dim=0)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)

        return x
