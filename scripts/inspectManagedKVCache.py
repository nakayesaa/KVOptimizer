import argparse
import csv
import os
import sys

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache import FullKVCache, StreamingKVCache, H2OKVCache, HybridKVCache, cacheSequenceLength, cacheSizeMb


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
        return HybridKVCache(budget=budget, sinkTokens=sinkTokens, recentTokens=recentTokens)
    raise ValueError(f"Unknown cacheType: {cacheType}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cacheType", choices=["full", "streaming", "h2o", "hybrid"], default="streaming")
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--sinkTokens", type=int, default=4)
    parser.add_argument("--recentTokens", type=int, default=8)
    parser.add_argument("--maxTokens", type=int, default=100)
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of artificial intelligence will likely transform every industry.",
    )
    arguments = parser.parse_args()

    os.makedirs(artifactDirectory, exist_ok=True)
    csvPath = os.path.join(artifactDirectory, f"{arguments.cacheType}CacheGrowth.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cacheManager = buildCache(
        arguments.cacheType,
        arguments.budget,
        arguments.sinkTokens,
        arguments.recentTokens,
    )

    tokenizer = GPT2Tokenizer.from_pretrained(modelName)
    modelArguments = {"attn_implementation": "eager"} if cacheManager.requiresAttentions else {}
    model = GPT2LMHeadModel.from_pretrained(modelName, **modelArguments)
    model.to(device).eval()

    inputIds = tokenizer(arguments.prompt, return_tensors="pt")["input_ids"].to(device)
    pastKeyValues = None
    tokensProcessed = 0

    with open(csvPath, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "cacheType", "sequenceLength", "cacheMb"])

        for step in range(arguments.maxTokens):
            inputLength = inputIds.shape[-1]
            positionIds = torch.arange(
                tokensProcessed,
                tokensProcessed + inputLength,
                device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                outputs = model(
                    input_ids=inputIds,
                    position_ids=positionIds,
                    past_key_values=pastKeyValues,
                    use_cache=True,
                    output_attentions=cacheManager.requiresAttentions,
                    return_dict=True,
                )

            pastKeyValues = cacheManager.manage(
                outputs.past_key_values,
                outputs.attentions if cacheManager.requiresAttentions else None,
            )
            tokensProcessed += inputLength

            sequenceLength = cacheSequenceLength(pastKeyValues)
            totalCacheMb = cacheSizeMb(pastKeyValues)
            writer.writerow([step + 1, arguments.cacheType, sequenceLength, totalCacheMb])

            print(
                f"Step {step + 1:03d} | cache_type={arguments.cacheType:<9} | "
                f"seq_len={sequenceLength:>4} | cache={totalCacheMb:.4f} MB"
            )

            inputIds = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

    print(f"\nSaved: {csvPath}")


if __name__ == "__main__":
    main()
