from enum import Enum, auto
from typing import Any

from models.base import BaseNet
from models.edge_gat import EdgeGAT
from models.edge_gcn import EdgeGCN
from models.gcn import GCN, GCN_MP


class ModelType(str, Enum):
    GCN = "GCN"
    GCN_MP = "GCN_MP"
    EDGE_GCN = "EDGE_GCN"
    EDGE_GAT = "EDGE_GAT"

    def __repr__(self):
        return self.value


def create_model(model_type: ModelType, num_nodes: int, edge_dim: int, target_size: int, device: Any) -> BaseNet:
    match model_type:
        case ModelType.GCN:
            model = GCN(embed_dim=num_nodes, target_size=target_size)
        case ModelType.GCN_MP:
            model = GCN_MP(embed_dim=num_nodes, target_size=target_size)
        case ModelType.EDGE_GCN:
            model = EdgeGCN(num_nodes=num_nodes, num_edge_types=edge_dim, target_size=target_size)
        case ModelType.EDGE_GAT:
            model = EdgeGAT(num_nodes=num_nodes, num_edge_types=edge_dim, target_size=target_size)
            pass
        case _:
            raise TypeError(f"Unsupported model: {model_type}")

    return model.to(device)
