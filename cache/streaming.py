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
        if layerIndex not in self.storeKeyCache:
            self.storeKeyCache[layerIndex] = keyState
            self.storeValueCache[layerIndex] = valueState
        else:
            self.storeKeyCache[layerIndex] = torch.cat(
                [self.storeKeyCache[layerIndex], keyState], dim=-2
            )
            self.storeValueCache[layerIndex] = torch.cat(
                [self.storeValueCache[layerIndex], valueState], dim=-2
            )
        currentLen = self.storeKeyCache[layerIndex].shape[-2]
        if currentLen > self.budget:
            sinkK = self.storeKeyCache[layerIndex][..., :self.keepSinkTokens, :]
            sinkV = self.storeValueCache[layerIndex][..., :self.keepSinkTokens, :]

            windowK = self.storeKeyCache[layerIndex][..., -self.holdWindowTokens:, :]
            windowV = self.storeValueCache[layerIndex][..., -self.holdWindowTokens:, :]
            
            self.storeKeyCache[layerIndex] = torch.cat([sinkK, windowK], dim=-2)
            self.storeValueCache[layerIndex] = torch.cat([sinkV, windowV], dim=-2)

        return self.storeKeyCache[layerIndex], self.storeValueCache[layerIndex]

    def get(self, layerIndex: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        if layerIndex not in self.storeKeyCache:
            return None
        return self.storeKeyCache[layerIndex], self.storeValueCache[layerIndex]

    def reset(self):
        self.storeKeyCache.clear()
        self.storeValueCache.clear()