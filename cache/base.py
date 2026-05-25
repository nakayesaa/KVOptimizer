import torch
from abc import ABC, abstractmethod

class BaseKVCache(ABC):
    "ABC for custom KV Cache implementations"
    def __init__(self, budget: int):
        self.budget = budget

    @abstractmethod
    def update(
        self,
        keyState: torch.Tensor,
        valueState: torch.Tensor,
        layerIndex: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        "Updates the cache with the new key and value states for a specific layer"
        pass

    @abstractmethod
    def get(self, layerIndex: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        "Retrieves the cached key and value states for a specific layer"
        pass

    @abstractmethod
    def reset(self):
        "Resets or clears the cache state"
        pass

