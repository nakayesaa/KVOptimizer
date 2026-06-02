from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


LegacyCache = tuple[tuple[torch.Tensor, torch.Tensor], ...]


def unpackPastKeyValues(pastKeyValues: Any) -> tuple[LegacyCache, str]:
    if isinstance(pastKeyValues, (tuple, list)):
        return tuple((keyState, valueState) for keyState, valueState in pastKeyValues), "legacy"

    if hasattr(pastKeyValues, "layers"):
        cacheLayers = []
        for layer in pastKeyValues.layers:
            keyState = getattr(layer, "keys", None)
            valueState = getattr(layer, "values", None)
            if keyState is None or valueState is None:
                raise TypeError("Cache layer has not been initialized with keys and values.")
            cacheLayers.append((keyState, valueState))
        return tuple(cacheLayers), "layers"

    if hasattr(pastKeyValues, "key_cache") and hasattr(pastKeyValues, "value_cache"):
        return tuple(zip(pastKeyValues.key_cache, pastKeyValues.value_cache)), "keyCache"

    if hasattr(pastKeyValues, "to_legacy_cache"):
        return tuple(pastKeyValues.to_legacy_cache()), "convertible"

    raise TypeError(
        "Unsupported past_key_values format. Expected a legacy tuple or a Hugging Face Cache object."
    )


def restorePastKeyValues(
    originalPastKeyValues: Any,
    managedLayers: LegacyCache,
    cacheLayout: str,
) -> Any:
    """Return pruned states in the same cache representation produced by the model."""
    if cacheLayout == "legacy":
        return managedLayers

    if cacheLayout == "layers":
        for layerIndex, (keyState, valueState) in enumerate(managedLayers):
            layer = originalPastKeyValues.layers[layerIndex]
            layer.keys = keyState
            layer.values = valueState
        return originalPastKeyValues

    if cacheLayout == "keyCache":
        for layerIndex, (keyState, valueState) in enumerate(managedLayers):
            originalPastKeyValues.key_cache[layerIndex] = keyState
            originalPastKeyValues.value_cache[layerIndex] = valueState
        return originalPastKeyValues

    if cacheLayout == "convertible":
        cacheType = type(originalPastKeyValues)
        if hasattr(cacheType, "from_legacy_cache"):
            return cacheType.from_legacy_cache(managedLayers)
        raise TypeError(
            "This Transformers Cache can be read but cannot be rebuilt after pruning. "
            "Use a Transformers version exposing Cache layers or from_legacy_cache()."
        )

    raise ValueError(f"Unknown cache layout: {cacheLayout}")


def cacheSequenceLength(pastKeyValues: Any) -> int:
    cacheLayers, _ = unpackPastKeyValues(pastKeyValues)
    if not cacheLayers:
        return 0
    return int(cacheLayers[0][0].shape[-2])


def cacheSizeMb(pastKeyValues: Any) -> float:
    cacheLayers, _ = unpackPastKeyValues(pastKeyValues)
    sizeBytes = sum(
        keyState.numel() * keyState.element_size()
        + valueState.numel() * valueState.element_size()
        for keyState, valueState in cacheLayers
    )
    return sizeBytes / (1024 ** 2)


class BaseKVCache(ABC):
    """Base policy for pruning the native cache returned by GPT-2."""

    requiresAttentions = False
    cacheType = "base"

    def __init__(self, budget: int | None):
        if budget is not None and budget <= 0:
            raise ValueError("budget must be positive.")
        self.budget = budget
        self.reset()

    def reset(self) -> None:
        self.cachedPositions = torch.empty(0, dtype=torch.long)
        self.totalTokensSeen = 0
        self.lastKeepIndices = torch.empty(0, dtype=torch.long)

    def getCachedPositions(self) -> list[int]:
        return self.cachedPositions.tolist()

    def manage(self, pastKeyValues: Any, attentions: Any = None) -> Any:
        """
        Prune a cache returned by the model.

        The caller must feed this returned managed cache into the next model
        forward pass. Do not append outputs.past_key_values again: the model
        already included prior K/V states in its output.
        """
        cacheLayers, cacheLayout = unpackPastKeyValues(pastKeyValues)
        if not cacheLayers:
            return pastKeyValues

        currentLength = int(cacheLayers[0][0].shape[-2])
        currentPositions = self._buildCurrentPositions(currentLength)
        self.beforeSelection(currentLength, attentions)

        keepIndices = self.selectIndices(currentPositions).to(dtype=torch.long, device="cpu")
        keepIndices = torch.unique(keepIndices, sorted=True)

        if keepIndices.numel() == 0:
            raise RuntimeError("A KV cache policy cannot evict every token.")
        if self.budget is not None and keepIndices.numel() > self.budget:
            raise RuntimeError("Cache policy returned more tokens than its budget.")

        managedLayers = tuple(
            (
                keyState.index_select(-2, keepIndices.to(keyState.device)),
                valueState.index_select(-2, keepIndices.to(valueState.device)),
            )
            for keyState, valueState in cacheLayers
        )

        self.afterSelection(keepIndices)
        self.cachedPositions = currentPositions.index_select(0, keepIndices)
        self.lastKeepIndices = keepIndices

        return restorePastKeyValues(pastKeyValues, managedLayers, cacheLayout)

    def _buildCurrentPositions(self, currentLength: int) -> torch.Tensor:
        cachedLength = int(self.cachedPositions.numel())

        if cachedLength == 0:
            self.totalTokensSeen = currentLength
            return torch.arange(currentLength, dtype=torch.long)

        newTokenCount = currentLength - cachedLength
        if newTokenCount < 0:
            raise RuntimeError(
                "The returned native cache is shorter than the managed cache. "
                "Reset the policy before starting a new sequence."
            )

        newPositions = torch.arange(
            self.totalTokensSeen,
            self.totalTokensSeen + newTokenCount,
            dtype=torch.long,
        )
        self.totalTokensSeen += newTokenCount
        return torch.cat([self.cachedPositions, newPositions], dim=0)

    def beforeSelection(self, currentLength: int, attentions: Any = None) -> None:
        return None

    def afterSelection(self, keepIndices: torch.Tensor) -> None:
        return None

    @abstractmethod
    def selectIndices(self, currentPositions: torch.Tensor) -> torch.Tensor:
        """Return chronological indices to retain from the model-returned cache."""
        raise NotImplementedError
