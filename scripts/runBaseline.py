import os
import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.perplexity import evaluatePerplexity
from benchmark.memory     import benchmarkMemoryAndThroughput

modelName         = "gpt2-medium"
artifactDirectory = "artifacts"
outputPath        = os.path.join(artifactDirectory, "baselineResults.json")
genSteps          = 200
maxPplWindows     = 50

os.makedirs(artifactDirectory, exist_ok=True)

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        totalVram = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
        print(f"VRAM: {totalVram:.0f} MB total")

    print(f"Model: {modelName}")
    tokenizer = GPT2Tokenizer.from_pretrained(modelName)
    model     = GPT2LMHeadModel.from_pretrained(modelName)
    model.to(device).eval()
    modelParams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params: {modelParams:.1f}M")

    results = {
        "model":    modelName,
        "device":   str(device),
        "cacheType": "full",
    }
    print("Benchmark 1 : Memory & Throughput")
    memResults = benchmarkMemoryAndThroughput(
        model=model,
        tokenizer=tokenizer,
        device=device,
        genSteps=genSteps,
    )
    results.update(memResults)

    print("Benchmark 2 : Perplexity (WikiText-2)")
    perplexity = evaluatePerplexity(
        model=model,
        tokenizer=tokenizer,
        device=device,
        maxSamples=maxPplWindows,
    )
    results["perplexity"] = round(perplexity, 4)
    with open(outputPath, "w") as f:
        json.dump(results, f, indent=2)

    print("Baseline Results Summary")
    print(f"{'Perplexity':<22} {results['perplexity']:>12.4f}")
    print(f"{'Peak VRAM (MB)':<22} {results['peakVramMb']:>12.2f}")
    print(f"{'Throughput (tok/s)':<22} {results['tokensSec']:>12.2f}")
    print(f"{'Total time (s)':<22} {results['totalTimeSec']:>12.2f}")
    print(f"{'Gen steps':<22} {results['genSteps']:>12}")
if __name__ == "__main__":
    main()
