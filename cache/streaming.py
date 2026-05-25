import torch
from .base import BaseKVCache

class StreamingKVCache(BaseKVCache):
    def __init__(self, budget: int, sinkTokens: int = 4):
        super().__init__(budget=budget)
        self.keepSinkTokens = sinkTokens
        self.holdWindowTokens = budget - sinkTokens
        self.storeKeyCache: dict[int, torch.Tensor] = {}
        self.storeValueCache: dict[int, torch.Tensor] = {}

    def update(
        self,
        keyState: torch.Tensor,
        valueState: torch.Tensor,
        layerIndex: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def get(self, layerIndex: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        if layerIndex not in self.storeKeyCache:
            return None
        return self.storeKeyCache[layerIndex], self.storeValueCache[layerIndex]

    def reset(self):
        self.storeKeyCache.clear()
        self.storeValueCache.clear()