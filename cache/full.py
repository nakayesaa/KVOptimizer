import torch

from .base import BaseKVCache


class FullKVCache(BaseKVCache):
    cacheType = "full"

    def __init__(self):
        super().__init__(budget=None)

    def selectIndices(self, currentPositions: torch.Tensor) -> torch.Tensor:
        return torch.arange(currentPositions.numel(), dtype=torch.long)
