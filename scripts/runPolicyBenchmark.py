import argparse
import json
import os
import sys

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.memory import benchmarkMemoryAndThroughput
from benchmark.perplexity import evaluatePolicyPerplexity
from cache import FullKVCache, StreamingKVCache, H2OKVCache, HybridKVCache


modelName = "gpt2-medium"
artifactDirectory = "artifacts"


def buildCache(cacheType: str, budget: int, sinkTokens: int, recentTokens: int):
    if cacheType == "full":
        return FullKVCache()
    if cacheType == "streaming":
        return StreamingKVCache(budget=budget, sinkTokens=sinkTokens)
    if cacheType == "h2o":
        return H2OKVCache(budget=budget, sinkTokens=sinkTokens)
    if cacheType == "hybrid":
        return HybridKVCache(
            budget=budget,
            sinkTokens=sinkTokens,
            recentTokens=recentTokens,
        )
    raise ValueError(f"Unknown cacheType: {cacheType}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cacheType", choices=["all", "full", "streaming", "h2o", "hybrid"], default="all")
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--sinkTokens", type=int, default=4)
    parser.add_argument("--recentTokens", type=int, default=32)
    parser.add_argument("--genSteps", type=int, default=200)
    parser.add_argument("--pplSeqLen", type=int, default=512)
    parser.add_argument("--pplWindows", type=int, default=10)
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {modelName}")

    tokenizer = GPT2Tokenizer.from_pretrained(modelName)
    model = GPT2LMHeadModel.from_pretrained(modelName, attn_implementation="eager")
    model.to(device).eval()

    policyNames = (
        ["full", "streaming", "h2o", "hybrid"]
        if arguments.cacheType == "all"
        else [arguments.cacheType]
    )
    results = []

    for cacheType in policyNames:
        print(f"\n{'=' * 64}\nCache policy: {cacheType}\n{'=' * 64}")
        cacheManager = buildCache(
            cacheType,
            arguments.budget,
            arguments.sinkTokens,
            arguments.recentTokens,
        )

        print("Benchmark 1 : Memory & Throughput")
        memoryResults = benchmarkMemoryAndThroughput(
            model=model,
            tokenizer=tokenizer,
            device=device,
            genSteps=arguments.genSteps,
            cacheManager=cacheManager,
        )

        print("Benchmark 2 : Policy Perplexity (WikiText-2)")
        cacheManager.reset()
        perplexity = evaluatePolicyPerplexity(
            model=model,
            tokenizer=tokenizer,
            device=device,
            cacheManager=cacheManager,
            seqLen=arguments.pplSeqLen,
            maxSamples=arguments.pplWindows,
        )

        results.append({
            "model": modelName,
            "device": str(device),
            "cacheType": cacheType,
            "budget": None if cacheType == "full" else arguments.budget,
            "sinkTokens": arguments.sinkTokens if cacheType != "full" else None,
            "recentTokens": arguments.recentTokens if cacheType == "hybrid" else None,
            "perplexity": round(perplexity, 4),
            **memoryResults,
        })

    os.makedirs(artifactDirectory, exist_ok=True)
    outputPath = os.path.join(artifactDirectory, "policyBenchmarkResults.json")
    with open(outputPath, "w") as file:
        json.dump(results, file, indent=2)

    print("\nPolicy Benchmark Summary")
    print(f"{'Cache Type':<14} {'PPL':>10} {'VRAM MB':>12} {'Tok/s':>12} {'Final KV':>12}")
    print("-" * 64)
    for result in results:
        vram = "CPU N/A" if result["peakVramMb"] is None else f"{result['peakVramMb']:.2f}"
        print(
            f"{result['cacheType']:<14} {result['perplexity']:>10.4f} "
            f"{vram:>12} {result['tokensSec']:>12.2f} "
            f"{result['finalCacheLength']:>12}"
        )
    print(f"\nSaved: {outputPath}")


if __name__ == "__main__":
    main()
