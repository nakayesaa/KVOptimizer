from typing import Any

import torch

from .base import BaseKVCache


class H2OKVCache(BaseKVCache):

    requiresAttentions = True
    cacheType = "h2o"

    def __init__(
        self,
        budget: int,
        sinkTokens: int = 4,
        recentTokens: int = 1,
    ):
        if sinkTokens < 0 or recentTokens < 0:
            raise ValueError("sinkTokens and recentTokens cannot be negative.")
        if sinkTokens + recentTokens > budget:
            raise ValueError("sinkTokens + recentTokens must not exceed budget.")
        super().__init__(budget=budget)
        self.keepSinkTokens = sinkTokens
        self.keepRecentTokens = recentTokens
        self.attentionScores: torch.Tensor | None = None

    def reset(self) -> None:
        super().reset()
        self.attentionScores = None

    def beforeSelection(self, currentLength: int, attentions: Any = None) -> None:
        if attentions is None:
            raise ValueError(
                "H2OKVCache requires outputs.attentions. "
                "Load GPT-2 with attn_implementation='eager' and pass output_attentions=True."
            )

        validAttentions = [attention for attention in attentions if attention is not None]
        if not validAttentions:
            raise ValueError("No attention tensors were returned by the model.")

        device = validAttentions[0].device
        currentScores = torch.zeros(currentLength, dtype=torch.float32, device=device)

        if self.attentionScores is not None:
            previousLength = self.attentionScores.numel()
            if previousLength > currentLength:
                raise RuntimeError("Attention score state is longer than the returned cache.")
            currentScores[:previousLength] = self.attentionScores.to(device)

        receivedScores = torch.zeros_like(currentScores)
        for attention in validAttentions:
            if attention.shape[-1] != currentLength:
                raise RuntimeError("Attention key length does not match returned cache length.")
            receivedScores += attention.float().sum(dim=-2).mean(dim=(0, 1))

        receivedScores /= len(validAttentions)
        self.attentionScores = currentScores + receivedScores

    def selectIndices(self, currentPositions: torch.Tensor) -> torch.Tensor:
        currentLength = currentPositions.numel()
        if currentLength <= self.budget:
            return torch.arange(currentLength, dtype=torch.long)

        if self.attentionScores is None:
            raise RuntimeError("H2O attention scores must be accumulated before selection.")

        protected = torch.zeros(currentLength, dtype=torch.bool)
        protected |= currentPositions < self.keepSinkTokens
        if self.keepRecentTokens > 0:
            protected[-self.keepRecentTokens:] = True

        protectedIndices = torch.nonzero(protected, as_tuple=False).flatten()
        openSlots = self.budget - protectedIndices.numel()

        candidateIndices = torch.nonzero(~protected, as_tuple=False).flatten()
        if openSlots > 0:
            candidateScores = self.attentionScores.index_select(
                0, candidateIndices.to(self.attentionScores.device)
            )
            topLocalIndices = torch.topk(
                candidateScores,
                k=min(openSlots, candidateIndices.numel()),
                largest=True,
            ).indices.cpu()
            selectedHeavyHitters = candidateIndices.index_select(0, topLocalIndices)
            keepIndices = torch.cat([protectedIndices, selectedHeavyHitters], dim=0)
        else:
            keepIndices = protectedIndices

        return torch.sort(keepIndices).values

    def afterSelection(self, keepIndices: torch.Tensor) -> None:
        if self.attentionScores is not None:
            self.attentionScores = self.attentionScores.index_select(
                0, keepIndices.to(self.attentionScores.device)
            ).detach()
