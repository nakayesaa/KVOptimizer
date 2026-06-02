from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from cache.base import BaseKVCache


def loadWikiTextIds(tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    print("Load WikiText-2")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    fullText = "\n\n".join(dataset["text"])
    return tokenizer(fullText, return_tensors="pt").input_ids[0]


def evaluatePerplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    seqLen: int = 1024,
    stride: int = 512,
    maxSamples: int = None,
) -> float:
    """Original full-context sliding-window baseline, retained for Phase 1."""
    inputIds = loadWikiTextIds(tokenizer)
    totalLength = inputIds.size(0)

    negativeLogLikelihood = 0.0
    totalScoredTokens = 0
    windowCount = 0
    previousEnd = 0

    for beginIndex in range(0, totalLength - 1, stride):
        endIndex = min(beginIndex + seqLen, totalLength)
        targetLength = endIndex - previousEnd
        windowIds = inputIds[beginIndex:endIndex].unsqueeze(0).to(device)

        labels = windowIds.clone()
        labels[:, :-targetLength] = -100

        with torch.no_grad():
            outputs = model(input_ids=windowIds, labels=labels, return_dict=True)

        negativeLogLikelihood += outputs.loss.item() * targetLength
        totalScoredTokens += targetLength
        previousEnd = endIndex
        windowCount += 1

        if windowCount % 10 == 0:
            runningPerplexity = math.exp(negativeLogLikelihood / totalScoredTokens)
            print(
                f" Processed {windowCount} windows | "
                f"running ppl ≈ {runningPerplexity:.2f}",
                end="\r",
            )

        if maxSamples and windowCount >= maxSamples:
            break
        if endIndex == totalLength:
            break

    perplexity = math.exp(negativeLogLikelihood / totalScoredTokens)
    print(f"\nWindows evaluated: {windowCount}")
    print(f"Perplexity: {perplexity:.4f}")
    return perplexity


def evaluatePolicyPerplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    cacheManager: BaseKVCache,
    seqLen: int = 512,
    maxSamples: int = 10,
) -> float:
    """
    Token-by-token perplexity under the actual cache eviction policy.

    Each window resets cache state, then the policy controls the context
    available for every next-token prediction within that window.
    """
    if seqLen > model.config.n_positions:
        raise ValueError(
            f"seqLen must not exceed GPT-2 n_positions={model.config.n_positions}."
        )

    allInputIds = loadWikiTextIds(tokenizer)
    totalLength = allInputIds.size(0)

    negativeLogLikelihood = 0.0
    totalScoredTokens = 0
    windowCount = 0

    for beginIndex in range(0, totalLength - 1, seqLen):
        endIndex = min(beginIndex + seqLen, totalLength)
        windowIds = allInputIds[beginIndex:endIndex]
        if windowIds.numel() < 2:
            break

        cacheManager.reset()
        pastKeyValues: Any = None

        for tokenIndex in range(windowIds.numel() - 1):
            currentToken = windowIds[tokenIndex].reshape(1, 1).to(device)
            targetToken = windowIds[tokenIndex + 1].reshape(1).to(device)
            positionIds = torch.tensor([[tokenIndex]], dtype=torch.long, device=device)

            with torch.no_grad():
                outputs = model(
                    input_ids=currentToken,
                    position_ids=positionIds,
                    past_key_values=pastKeyValues,
                    use_cache=True,
                    output_attentions=cacheManager.requiresAttentions,
                    return_dict=True,
                )

            logits = outputs.logits[:, -1, :]
            negativeLogLikelihood += F.cross_entropy(logits, targetToken).item()
            totalScoredTokens += 1

            pastKeyValues = cacheManager.manage(
                outputs.past_key_values,
                outputs.attentions if cacheManager.requiresAttentions else None,
            )

        windowCount += 1
        runningPerplexity = math.exp(negativeLogLikelihood / totalScoredTokens)
        print(
            f" Processed {windowCount} policy windows | "
            f"running ppl ≈ {runningPerplexity:.2f}",
            end="\r",
        )

        if maxSamples and windowCount >= maxSamples:
            break

    perplexity = math.exp(negativeLogLikelihood / totalScoredTokens)
    print(f"\nPolicy windows evaluated: {windowCount}")
    print(f"Policy perplexity: {perplexity:.4f}")
    return perplexity
