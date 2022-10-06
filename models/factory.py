from enum import Enum
from functools import partial
from typing import Any

from torch.nn import RNN, LSTM

from models.base import BaseNet
from models.gcn_base import GCNBaseline, GCNBaselineEmb
from models.bert import Bert
from models.edge_gat import EdgeGAT
from models.edge_gcn import EdgeGCN
from models.feed_forward import FeedForward
from models.ff_positional_emb import FeedForwardPos
from models.rnn import SequenceNet


class ModelType(str, Enum):
    GCN_BASELINE = "GCN_BASELINE"
    GCN_BASELINE_EMB = "GCN_BASELINE_EMB"
    EDGE_GCN = "EDGE_GCN"
    EDGE_GAT = "EDGE_GAT"
    RNN = "RNN"
    BI_RNN = "BI_RNN"
    LSTM = "LSTM"
    FEED_FORWARD = "FEED_FORWARD"
    BERT = "BERT"
    FEED_FORWARD_POS = "FEED_FORWARD_POS"

    def __repr__(self):
        return self.value


def create_model(
    model_type: ModelType,
    num_nodes: int,
    edge_dim: int,
    target_size: int,
    relation_lst: list[str],
    device: Any,
    seq_embed_dim: int = 100,
) -> BaseNet:
    seq_net_factory = partial(
        SequenceNet,
        num_nodes=num_nodes,
        num_edge_types=edge_dim,
        target_size=target_size,
        relation_lst=relation_lst,
        embed_dim=seq_embed_dim,
    )
    seq_encoder_kwargs = {
        "input_size": seq_embed_dim,
        "hidden_size": 64,
        "batch_first": True,
    }

    match model_type:
        case ModelType.GCN_BASELINE:
            model = GCNBaseline(embed_dim=num_nodes, target_size=target_size)
        case ModelType.GCN_BASELINE_EMB:
            model = GCNBaselineEmb(num_nodes=num_nodes, target_size=target_size)
        case ModelType.EDGE_GCN:
            model = EdgeGCN(num_nodes=num_nodes, num_edge_types=edge_dim, target_size=target_size)
        case ModelType.EDGE_GAT:
            model = EdgeGAT(num_nodes=num_nodes, num_edge_types=edge_dim, target_size=target_size)
        case ModelType.RNN:
            model = seq_net_factory(
                encoder=RNN(
                    **seq_encoder_kwargs,
                    bidirectional=False
                )
            )
        case ModelType.BI_RNN:
            model = seq_net_factory(
                encoder=RNN(
                    **seq_encoder_kwargs,
                    bidirectional=True
                )
            )
        case ModelType.LSTM:
            model = seq_net_factory(
                encoder=LSTM(
                    **seq_encoder_kwargs,
                    bidirectional=False
                )
            )
        case ModelType.FEED_FORWARD:
            model = FeedForward(target_size=target_size, relation_lst=relation_lst)

        case ModelType.BERT:
            model = Bert(target_size=target_size)

        case ModelType.FEED_FORWARD_POS:
            model = FeedForwardPos(target_size=target_size, relation_lst=relation_lst)

        case _:
            raise TypeError(f"Unsupported model: {model_type}")

    return model.to(device)
