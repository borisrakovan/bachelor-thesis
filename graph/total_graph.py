import functools
from typing import Callable

import numpy as np
import torch
from torch_geometric.data import Data

from clutrr.preprocess import Instance
from graph.base import GraphFactory, Edge
from nlp.embeddings import WordEmbeddings
from nlp.tokenization import tokenize_normalize_naive, tokenize_normalize_wo_stopwords, tokenize_normalize_relevant


class _TotalGraph(GraphFactory):
    def __init__(self, entity_lst: list[str], relation_lst: list[str], batch_size: int, device: str, tokenize_func: Callable):
        super().__init__(entity_lst, relation_lst, batch_size, device)
        self.tokenize_func = tokenize_func
        self.embeddings = WordEmbeddings()

    @property
    def input_dim(self):
        return self.embeddings.embedding_dim

    @property
    def edge_dim(self) -> int:
        return -1

    def _construct_graph(self, instance: Instance, _: int) -> tuple[Data, Edge]:
        vocab = self.relation_lst
        story_tokens, story_entities = self.tokenize_func(instance, vocab=vocab)

        story_encoding: list[np.array] = []
        for token_index, token in enumerate(story_tokens):
            if token in story_entities:
                token_emb = self.embeddings.special_ent_embedding
            else:
                token_emb = self.embeddings.word_to_vec_map.get(token)
                if token_emb is None:
                    print(f"Missing embedding: {token}. Shouldn't happen.")
                    raise KeyError

            story_encoding.append(token_emb.tolist())

        edge_index = []
        for i in range(len(story_encoding)):
            for j in range(len(story_encoding)):
                # total graph
                edge_index.append([i, j])

        tgt_frm, tgt_rel, tgt_to = instance.target
        try:
            target_edge = (
                next(idx for idx, token in enumerate(story_tokens) if token == tgt_frm.lower()),
                next(idx for idx, token in enumerate(story_tokens) if token == tgt_to.lower())
            )
        except StopIteration:
            print(story_tokens)
            print(story_entities)
            print(instance.raw_story)
            print(instance.target)
            raise

        y = self.relation_to_idx[tgt_rel]

        data = Data(
            x=torch.tensor(story_encoding, device=self.device),
            edge_index=torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous(),
            # edge_attr=torch.tensor(edge_attr, dtype=torch.long, device=device).view(-1, 1),
            y=torch.tensor([y], device=self.device).view(-1, 1)
        )
        return data, target_edge


TotalGraphV1 = functools.partial(_TotalGraph, tokenize_func=tokenize_normalize_naive)
TotalGraphV2 = functools.partial(_TotalGraph, tokenize_func=tokenize_normalize_wo_stopwords)
TotalGraphV3 = functools.partial(_TotalGraph, tokenize_func=tokenize_normalize_relevant)