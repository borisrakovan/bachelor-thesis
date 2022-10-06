from dataclasses import dataclass
from typing import Any, Type, TypedDict

from graph.base import GraphFactory
from models.factory import ModelType


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    num_epochs: int
    lr: float
    evaluate_every: int = 10


@dataclass(frozen=True)
class Experiment:
    model_type: ModelType
    graph_factory_cls: Type[GraphFactory]
    train_config: TrainConfig
    num_training_samples: int = -1


class TrainHistory(TypedDict):
    train_losses: list[float]
    train_acc: list[float]
    test_acc: dict[str, list[float]]
