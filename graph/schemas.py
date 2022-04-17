from dataclasses import dataclass
from typing import Any

from torch import Tensor

from clutrr.preprocess import Instance


@dataclass
class InputBatch:
    geo_batch: Any  # Batch
    targets: Tensor
    instances: list[Instance]

