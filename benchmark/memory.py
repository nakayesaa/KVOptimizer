import time
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Dict, Any


def benchmarkMemoryAndThroughput(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    prompt: str = "The future of artificial intelligence is",
    genSteps: int = 200,
) -> Dict[str, Any]:
    inputIds      = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    pastKeyValues = None

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    startTime = time.perf_counter()
    for step in range(genSteps):
        with torch.no_grad():
            outputs = model(
                input_ids=inputIds,
                past_key_values=pastKeyValues,
                use_cache=True,
                return_dict=True,
            )

        pastKeyValues = outputs.past_key_values
        nextToken = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        inputIds  = nextToken

        if (step + 1) % 50 == 0:
            elapsed = time.perf_counter() - startTime
            tps     = (step + 1) / elapsed
            seqLen  = pastKeyValues[0][0].shape[2]
            print(f"Step {step + 1:03d} | seq_len={seqLen} | {tps:.1f} tok/s", end="\r")

    torch.cuda.synchronize(device)
    totalTime = time.perf_counter() - startTime

    peakVramBytes = torch.cuda.max_memory_allocated(device)
    peakVramMb    = peakVramBytes / (1024 ** 2)
    tokensSec     = genSteps / totalTime

    torch.cuda.empty_cache()

    print(f"\nPeak VRAM: {peakVramMb:.2f} MB")
    print(f"Tokens/sec: {tokensSec:.2f}")
    print(f"Total time: {totalTime:.2f}s ({genSteps} tokens)")

    return {
        "peakVramMb":   round(peakVramMb, 2),
        "tokensSec":    round(tokensSec, 2),
        "totalTimeSec": round(totalTime, 2),
        "genSteps":     genSteps,
    }
