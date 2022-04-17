from abc import ABC, abstractmethod

from torch import Tensor, nn

from graph.schemas import InputBatch


class BaseNet(ABC, nn.Module):
    @abstractmethod
    def forward(self, input_batch: InputBatch) -> Tensor:
        ...
