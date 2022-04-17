import torch
from torch_geometric.data import Data

from clutrr.preprocess import Instance
from graph.base import GraphFactory, Edge


class KinshipGraph(GraphFactory):
    @property
    def input_dim(self) -> int:
        return len(self.entity_lst)

    @property
    def edge_dim(self) -> int:
        return len(self.relation_lst)

    def _construct_graph(self, instance: Instance, max_num_entities: int) -> tuple[Data, Edge]:
        entity_lst = sorted({x for t in instance.story for x in {t[0], t[2]}})
        entity_to_idx = {e: i for i, e in enumerate(entity_lst)}

        x = torch.arange(max_num_entities, device=self.device).view(-1, 1)

        edge_list = [(entity_to_idx[s], entity_to_idx[o]) for (s, _, o) in instance.story]
        edge_index = torch.tensor(list(zip(*edge_list)), dtype=torch.long, device=self.device)

        edge_types = [self.relation_to_idx[p] for (_, p, _) in instance.story]
        y = torch.tensor([self.relation_to_idx[instance.target[1]]], device=self.device).view(-1, 1)

        edge_attr = torch.tensor(edge_types, dtype=torch.long, device=self.device).view(-1, 1)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        target_edge = (entity_to_idx[instance.target[0]], entity_to_idx[instance.target[2]])
        return data, target_edge
