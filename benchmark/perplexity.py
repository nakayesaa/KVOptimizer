import torch
import math
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

def evaluatePerplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    seqLen: int = 1024,
    stride: int = 512,
    maxSamples: int = None,
) -> float:
    print("Load WikiText-2")
    dataset   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    fullText  = "\n\n".join(dataset["text"])

    encodings = tokenizer(fullText, return_tensors="pt")
    inputIds  = encodings.input_ids[0]
    totalLen  = inputIds.size(0)

    nlls        = []
    windowCount = 0
    prevEnd     = 0

    for beginIndex in range(0, totalLen - 1, stride):
        endIdx   = min(beginIndex + seqLen, totalLen)
        targetLen = endIdx - prevEnd

        windowIds = inputIds[beginIndex:endIdx].unsqueeze(0).to(device)

        labels = windowIds.clone()
        labels[:, :-targetLen] = -100
        with torch.no_grad():
            outputs = model(input_ids=windowIds, labels=labels, return_dict=True)
            nlls.append(outputs.loss.item() * targetLen)

        prevEnd = endIdx
        windowCount += 1

        if windowCount % 10 == 0:
            print(f" Processed {windowCount} windows | running ppl ≈ "
                  f"{math.exp(sum(nlls) / sum(range(stride, stride * windowCount + 1, stride))):.2f}",
                  end="\r")

        if maxSamples and windowCount >= maxSamples:
            break
        if endIdx == totalLen:
            break

    totalScoredTokens = prevEnd
    avgNll            = sum(nlls) / totalScoredTokens
    perplexity        = math.exp(avgNll)

    print(f"\nWindows evaluated: {windowCount}")
    print(f"Perplexity: {perplexity:.4f}")

    return perplexity
