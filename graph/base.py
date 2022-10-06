from abc import ABC, abstractmethod
from typing import Any

import torch
from torch_geometric.data import Batch, Data

from clutrr.preprocess import Instance
from graph.schemas import InputBatch
from utils import make_batches

Edge = tuple[int, int]


class GraphFactory(ABC):
    def __init__(
        self,
        entity_lst: list[str],
        relation_lst: list[str],
        batch_size: int,
        device: Any
    ):
        self.entity_lst = entity_lst
        self.relation_lst = relation_lst
        self.relation_to_idx = {rel: idx for idx, rel in enumerate(self.relation_lst)}

        self.batch_size = batch_size
        self.device = device

    @property
    @abstractmethod
    def input_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def edge_dim(self) -> int:
        ...

    @abstractmethod
    def _construct_graph(self, instance: Instance, max_num_entities: int) -> tuple[Data, Edge]:
        ...

    def create_batches(self, instances: list[Instance]) -> list[InputBatch]:
        num_instances = len(instances)
        batches = make_batches(num_instances, self.batch_size)

        res = []
        for i, (batch_start, batch_end) in enumerate(batches):
            if i > 0 and i % 40 == 0:
                print(f"Processed {i}/{len(batches)}")
            batch_instances = instances[batch_start:batch_end]
            max_num_entities = max(i.num_nodes for i in batch_instances)

            batch_data = [
                self._construct_graph(inst, max_num_entities)
                for inst in batch_instances
            ]

            geo_batch = Batch.from_data_list([x[0] for x in batch_data])
            raw_targets = [list(x[1]) for x in batch_data]

            targets = torch.tensor(raw_targets, dtype=torch.long, device=self.device)
            batch = InputBatch(geo_batch, targets, batch_instances)
            res.append(batch)

        return res
