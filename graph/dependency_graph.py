import spacy
import torch
from spacy import displacy
from spacy.tokens import Token
from torch_geometric.data import Data

from clutrr.preprocess import Instance
from graph.base import GraphFactory, Edge


class Vocabulary:
    def __init__(self, relation_lst: list):
        self._relation_to_id: dict[str, int] = {rel: i for i, rel in enumerate(relation_lst)}
        self._token_to_id: dict[str, int] = {}
        self._edge_type_to_id: dict[str, int] = {}

    @property
    def relation_to_id(self):
        return self._relation_to_id

    def token_to_id(self, tkn: str) -> int:
        if tkn not in self._token_to_id:
            self._token_to_id[tkn] = len(self._token_to_id)

        return self._token_to_id[tkn]

    def edge_type_to_id(self, typ: str) -> int:
        if typ not in self._edge_type_to_id:
            self._edge_type_to_id[typ] = len(self._edge_type_to_id)

        return self._edge_type_to_id[typ]

    @property
    def num_tokens(self):
        return len(self._token_to_id)

    @property
    def num_edge_types(self):
        return len(self._edge_type_to_id)


class DependencyGraph(GraphFactory):
    def __init__(self, entity_lst: list[str], relation_lst: list[str], batch_size: int, device: str):
        super().__init__(entity_lst, relation_lst, batch_size, device)
        self.vocab = Vocabulary(relation_lst)
        self.nlp = spacy.load("en_core_web_sm")

    @property
    def input_dim(self) -> int:
        return self.vocab.num_tokens

    @property
    def edge_dim(self) -> int:
        return self.vocab.num_edge_types

    def _construct_graph(self, instance: Instance, max_num_entities: int) -> tuple[Data, Edge]:
        # todo: more intelligent graph creation
        #  spacy already has embeddings - start with those?
        #  keep only alpha tokens, then only certain POS
        story = instance.raw_story.replace("[", "").replace("]", "")
        doc = self.nlp(story)
        # todo visualize
        # displacy.serve(doc, style="dep")

        tokens: list[Token] = list(doc)

        # num_features is 1, and it is equal to the token encoding in the vocabulary
        x = [self.vocab.token_to_id(tkn.text) for tkn in tokens]

        token_to_idx = {tkn.text: i for i, tkn in enumerate(tokens)}

        # for token in doc:
        #     parts = [token.text, token.dep_, token.head.text, token.head.pos_, str([child for child in token.children])]
        # print(" | ".join(parts))

        edge_index_t = []
        edge_attr = []
        self_loops = 0
        for i, token in enumerate(tokens):
            for child in token.children:
                frm, to = token_to_idx[token.text], token_to_idx[child.text]
                if frm == to:
                    self_loops += 1
                    continue
                edge_index_t.append([frm, to])
                edge_typ_id = self.vocab.edge_type_to_id(child.dep_)
                # edge feature is the id of the edge type
                edge_attr.append(edge_typ_id)

        edge_index = [[edge[0] for edge in edge_index_t], [edge[1] for edge in edge_index_t]]

        # todo: we treat relations as labels and relations in text differently here!

        try:
            target_edge = (token_to_idx[instance.target[0]], token_to_idx[instance.target[2]])
        except KeyError:
            print(instance.target)
            print(instance.raw_story)
            print(token_to_idx)
            raise

        target_rel = self.vocab.relation_to_id[instance.target[1]]

        return Data(
            x=torch.tensor(x, dtype=torch.long, device=self.device).view(-1, 1),
            edge_index=torch.tensor(edge_index, dtype=torch.long, device=self.device),
            edge_attr=torch.tensor(edge_attr, dtype=torch.long, device=self.device).view(-1, 1),
            y=torch.tensor([target_rel], device=self.device).view(-1, 1)
        ), target_edge
