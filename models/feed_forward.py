from abc import ABC, abstractmethod
from collections import Counter

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from clutrr.preprocess import Instance
from graph.schemas import InputBatch
from models.base import BaseNet
from nlp.embeddings import WordEmbeddings


class FeedForwardVocab(ABC):
    UNK_TOKEN = "UNK"

    def __init__(self, relation_lst: list[str]):
        self.relation_lst = relation_lst
        self.token_to_idx = {rel: idx for idx, rel in enumerate(self.relation_lst)}

    @property
    @abstractmethod
    def num_tokens(self):
        pass

    @abstractmethod
    def encode_batch(self, instances: list[Instance]):
        pass

    def register_parameters(self, net: BaseNet):
        pass


class RelationVocab(FeedForwardVocab):
    def encode(self, instance: Instance) -> list[int]:
        # encode only the relations
        seq = [t[1] for t in instance.story]
        tokens = [self.token_to_idx[s] for s in seq]

        counts = Counter(tokens)
        bag_of_relations = [
            counts[i] for i in range(self.num_tokens)
        ]
        return bag_of_relations

    @property
    def num_tokens(self):
        return len(self.token_to_idx)

    def encode_batch(self, instances: list[Instance]):
        batch = []
        for instance in instances:
            batch.append(self.encode(instance))

        return torch.tensor(batch, dtype=torch.float)


class RelationEmbVocab(FeedForwardVocab):
    def __init__(self, relation_lst: list[str], embed_dim: int = 20):
        super().__init__(relation_lst)
        self.emb = WordEmbeddings()

    @property
    def num_tokens(self):
        return self.emb.embedding_dim

    def encode(self, instance: Instance) -> torch.tensor:
        # encode only the relations
        seq = [t[1] for t in instance.story]
        # tokens = [self.token_to_idx[s] for s in seq]

        counts = Counter(seq)

        emb = torch.zeros((len(seq), self.emb.embedding_dim))

        for idx, token in enumerate(seq):
            token_emb = self.emb.word_to_vec_map[token]
            emb[idx, :] = torch.from_numpy(token_emb) * counts[token]

        return torch.mean(emb, dim=0)

    def encode_batch(self, instances: list[Instance]):
        batch = []
        for instance in instances:
            batch.append(self.encode(instance).unsqueeze(0))

        return torch.cat(batch, dim=0)

    def register_parameters(self, net: BaseNet):
        # net.register_parameter("token_embeddings", nn.Parameter(self.token_embeddings.weight))
        pass


class FeedForward(BaseNet):
    def __init__(self, target_size: int, relation_lst: list[str], hidden_size: int = 32):
        super().__init__()
        self.vocab = RelationVocab(relation_lst=relation_lst)
        self.lin1 = nn.Linear(in_features=self.vocab.num_tokens, out_features=hidden_size)
        self.lin2 = nn.Linear(in_features=hidden_size, out_features=target_size)

        self.vocab.register_parameters(self)

    def forward(self, input_batch: InputBatch) -> Tensor:
        device = input_batch.targets.device

        x = self.vocab.encode_batch(input_batch.instances).to(device)

        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)

        return x
