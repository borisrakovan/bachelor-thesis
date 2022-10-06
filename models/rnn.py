import torch
from torch import Tensor, nn

from clutrr.preprocess import Instance
from graph.schemas import InputBatch
from models.base import BaseNet
from utils import pad_sequences


class EntityRelVocab:
    UNK_TOKEN = "UNK"

    def __init__(self, num_entities: int, relation_lst: list[str]):
        self.entity_lst = [f"ENTITY_{i}" for i in range(num_entities)]
        self.relation_lst = relation_lst

        self.token_lst = sorted({s for s in self.entity_lst + self.relation_lst} | {self.UNK_TOKEN})
        self.token_to_idx = {s: i for i, s in enumerate(self.token_lst)}

    @property
    def num_tokens(self):
        return len(self.token_to_idx)

    def encode(self, instance: Instance) -> list[int]:
        entity_lst = sorted({e for t in instance.story + [instance.target] for e in {t[0], t[2]}})
        entity_to_symbol = {e: f"ENTITY_{i}" for i, e in enumerate(entity_lst)}
        new_story = [(entity_to_symbol[t[0]], t[1], entity_to_symbol[t[2]]) for t in instance.story]
        new_target = (
            entity_to_symbol[instance.target[0]],
            instance.target[1],
            entity_to_symbol[instance.target[2]]
        )
        seq = [new_target] + new_story

        # test_data["1.10_test.csv"][0].instances[1].story
        return [self.token_to_idx[s] for t in seq for s in t]

    def encode_2(self, instance: Instance) -> list[int]:
        # encode only the relations
        seq = [t[1] for t in instance.story]

        if instance.target[0] != instance.story[0][0] or instance.target[2] != instance.story[-1][2]:
            print(instance)
            print(seq)
            assert False
        # test_data["1.10_test.csv"][0].instances[1].story
        return [self.token_to_idx[s] for s in seq]

    def encode_batch(self, instances: list[Instance]):
        linear_story_lst = []

        for instance in instances:
            linear_story_lst.append(self.encode(instance))

        return pad_sequences(linear_story_lst, value=self.token_to_idx[self.UNK_TOKEN])


class RelationVocab:
    UNK_TOKEN = "UNK"

    def __init__(self, relation_lst: list[str]):
        self.relation_lst = relation_lst
        token_lst = enumerate(self.relation_lst + [self.UNK_TOKEN])
        self.token_to_idx = {rel: idx for idx, rel in token_lst}

    @property
    def num_tokens(self):
        return len(self.token_to_idx)

    def encode(self, instance: Instance) -> list[int]:
        # encode only the relations
        seq = [t[1] for t in instance.story]
        return [self.token_to_idx[s] for s in seq]

    def encode_batch(self, instances: list[Instance]):
        linear_story_lst = []

        for instance in instances:
            linear_story_lst.append(self.encode(instance))

        return pad_sequences(linear_story_lst, value=self.token_to_idx[self.UNK_TOKEN])


class SequenceNet(BaseNet):

    def __init__(
        self,
        encoder: nn.Module,
        num_nodes: int,
        num_edge_types: int,
        target_size: int,
        relation_lst: list[str],
        embed_dim: int,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edge_types = num_edge_types
        self.relation_lst = relation_lst
        self.target_size = target_size

        # self.vocab = RelationVocab(relation_lst=relation_lst)
        self.vocab = EntityRelVocab(num_entities=num_nodes, relation_lst=relation_lst)

        self.token_embeddings = nn.Embedding(self.vocab.num_tokens, embed_dim)
        nn.init.xavier_uniform_(self.token_embeddings.weight)

        self.encoder = encoder

        self.projection = None

    def forward(self, input_batch: InputBatch) -> Tensor:
        device = input_batch.targets.device
        sequences = self.vocab.encode_batch(input_batch.instances)

        batch_seq = torch.tensor(sequences, dtype=torch.long, device=device)
        batch_seq_emb = self.token_embeddings(batch_seq).to(device)

        out, hidden = self.encoder(batch_seq_emb)
        # out, hidden = self.encoder(batch_seq.unsqueeze(-1).float())

        last = out[:, -1, :]

        if self.projection is None:
            in_dim = out.shape[-1]
            self.projection = nn.Linear(in_dim, self.target_size).to(device)

        return self.projection(last)
