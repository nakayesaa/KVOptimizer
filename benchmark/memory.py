from __future__ import annotations

import time
from typing import Any, Dict

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from cache.base import BaseKVCache, cacheSequenceLength


def benchmarkMemoryAndThroughput(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    prompt: str = "The future of artificial intelligence is",
    genSteps: int = 200,
    cacheManager: BaseKVCache | None = None,
) -> Dict[str, Any]:
    inputIds = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    pastKeyValues = None
    tokensProcessed = 0

    if cacheManager is not None:
        cacheManager.reset()

    usingCuda = device.type == "cuda"
    if usingCuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    startTime = time.perf_counter()

    for step in range(genSteps):
        inputLength = inputIds.shape[-1]
        if tokensProcessed + inputLength > model.config.n_positions:
            raise ValueError(
                f"GPT-2 supports at most {model.config.n_positions} absolute positions. "
                "Reduce prompt length or genSteps."
            )

        modelArguments = {
            "input_ids": inputIds,
            "past_key_values": pastKeyValues,
            "use_cache": True,
            "return_dict": True,
        }

        if cacheManager is not None:
            modelArguments["position_ids"] = torch.arange(
                tokensProcessed,
                tokensProcessed + inputLength,
                device=device,
            ).unsqueeze(0)
            modelArguments["output_attentions"] = cacheManager.requiresAttentions

        with torch.no_grad():
            outputs = model(**modelArguments)

        pastKeyValues = outputs.past_key_values
        if cacheManager is not None:
            pastKeyValues = cacheManager.manage(
                pastKeyValues,
                outputs.attentions if cacheManager.requiresAttentions else None,
            )

        tokensProcessed += inputLength
        nextToken = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        inputIds = nextToken

        if (step + 1) % 50 == 0:
            elapsed = time.perf_counter() - startTime
            tokensSec = (step + 1) / elapsed
            sequenceLength = cacheSequenceLength(pastKeyValues)
            print(
                f"Step {step + 1:03d} | seq_len={sequenceLength} | "
                f"{tokensSec:.1f} tok/s",
                end="\r",
            )

    if usingCuda:
        torch.cuda.synchronize(device)
    totalTime = time.perf_counter() - startTime

    peakVramMb = None
    if usingCuda:
        peakVramBytes = torch.cuda.max_memory_allocated(device)
        peakVramMb = peakVramBytes / (1024 ** 2)
        torch.cuda.empty_cache()

    tokensSec = genSteps / totalTime
    finalCacheLength = cacheSequenceLength(pastKeyValues)

    if peakVramMb is not None:
        print(f"\nPeak VRAM: {peakVramMb:.2f} MB")
    else:
        print("\nPeak VRAM: unavailable on CPU")
    print(f"Tokens/sec: {tokensSec:.2f}")
    print(f"Total time: {totalTime:.2f}s ({genSteps} tokens)")
    print(f"Final cache length: {finalCacheLength}")

    return {
        "peakVramMb": round(peakVramMb, 2) if peakVramMb is not None else None,
        "tokensSec": round(tokensSec, 2),
        "totalTimeSec": round(totalTime, 2),
        "genSteps": genSteps,
        "finalCacheLength": finalCacheLength,
    }
