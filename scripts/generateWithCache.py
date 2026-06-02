import argparse
import os
import sys

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache import FullKVCache, StreamingKVCache, H2OKVCache, HybridKVCache, cacheSequenceLength


modelName = "gpt2-medium"


def buildCache(
    cacheType: str,
    budget: int,
    sinkTokens: int,
    recentTokens: int,
):
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
    parser.add_argument("--cacheType", choices=["full", "streaming", "h2o", "hybrid"], default="streaming")
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--sinkTokens", type=int, default=4)
    parser.add_argument("--recentTokens", type=int, default=32)
    parser.add_argument("--maxTokens", type=int, default=200)
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of artificial intelligence will likely",
    )
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cacheManager = buildCache(
        arguments.cacheType,
        arguments.budget,
        arguments.sinkTokens,
        arguments.recentTokens,
    )

    tokenizer = GPT2Tokenizer.from_pretrained(modelName)
    modelArguments = {}
    if cacheManager.requiresAttentions:
        modelArguments["attn_implementation"] = "eager"
    model = GPT2LMHeadModel.from_pretrained(modelName, **modelArguments)
    model.to(device).eval()

    inputIds = tokenizer(arguments.prompt, return_tensors="pt")["input_ids"].to(device)
    promptLength = inputIds.shape[-1]
    if promptLength + arguments.maxTokens - 1 > model.config.n_positions:
        raise ValueError(
            f"Prompt plus generation exceeds GPT-2 limit of {model.config.n_positions} positions."
        )

    generatedIds = inputIds[0].cpu().tolist()
    pastKeyValues = None
    tokensProcessed = 0

    for _ in range(arguments.maxTokens):
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

        nextToken = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generatedIds.append(nextToken.item())
        inputIds = nextToken

    print(f"Cache type        : {arguments.cacheType}")
    print(f"Final cache length: {cacheSequenceLength(pastKeyValues)}")
    print("\nGenerated text:\n")
    print(tokenizer.decode(generatedIds, skip_special_tokens=True))


if __name__ == "__main__":
    main()
