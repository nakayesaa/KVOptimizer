import torch
from .base import BaseKVCache

class FullKVCache(BaseKVCache):
    def __init__(self, budget: int = None):
        super().__init__(budget=budget)
        self.KCache: dict[int, torch.Tensor] = {}
        self.VCache: dict[int, torch.Tensor] = {}
    def update(
        self,
        keyState: torch.Tensor,
        valueState: torch.Tensor,
        layerIndex: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layerIndex not in self.KCache:
            self.KCache[layerIndex] = keyState
            self.VCache[layerIndex] = valueState
        else:
            self.KCache[layerIndex] = torch.cat(
                [self.KCache[layerIndex], keyState], dim=-2
            )
            self.VCache[layerIndex] = torch.cat(
                [self.VCache[layerIndex], valueState], dim=-2
            )
        return self.KCache[layerIndex], self.VCache[layerIndex]

    def get(self, layerIndex: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        if layerIndex not in self.KCache:
            return None
        return self.KCache[layerIndex], self.VCache[layerIndex]

    def reset(self):
        self.KCache.clear()
        self.VCache.clear()