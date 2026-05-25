import torch
from abc import ABC, abstractmethod

class BaseKVCache(ABC):
    "ABC for custom KV Cache implementations"
    def __init__(self, budget: int):
        self.budget = budget
    @abstractmethod
    def update(self, key_state: torch.Tensor, value_state: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        "Updates the cache with the new key and value states for a specific layer"
        pass
    @abstractmethod
    def reset(self):
        "Resets or clears the cache state"
        pass
