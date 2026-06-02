import torch

from .base import BaseKVCache


class StreamingKVCache(BaseKVCache):

    cacheType = "streaming"

    def __init__(self, budget: int, sinkTokens: int = 4):
        if sinkTokens < 0:
            raise ValueError("sinkTokens cannot be negative.")
        if sinkTokens >= budget:
            raise ValueError("sinkTokens must be smaller than budget.")
        super().__init__(budget=budget)
        self.keepSinkTokens = sinkTokens
        self.holdWindowTokens = budget - sinkTokens

    def selectIndices(self, currentPositions: torch.Tensor) -> torch.Tensor:
        currentLength = currentPositions.numel()
        if currentLength <= self.budget:
            return torch.arange(currentLength, dtype=torch.long)

        sinkIndices = torch.nonzero(
            currentPositions < self.keepSinkTokens,
            as_tuple=False,
        ).flatten()
        windowIndices = torch.arange(
            currentLength - self.holdWindowTokens,
            currentLength,
            dtype=torch.long,
        )
        return torch.cat([sinkIndices, windowIndices], dim=0)
