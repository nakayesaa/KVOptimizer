import os
import csv
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

modelName          = "gpt2-medium"
maxTokens          = 20
artifactDirectory  = "artifacts"
attentionDirectory = os.path.join(artifactDirectory, "attentions")

os.makedirs(attentionDirectory, exist_ok=True)

def tensorSizeMb(tensor: torch.Tensor) -> float:
    return tensor.numel() * tensor.element_size() / (1024 ** 2)

def kvCacheSizeMb(pastKeyValues) -> float:
    sum = 0
    for k,v in pastKeyValues:
        sum = sum + tensorSizeMb(k) + tensorSizeMb(v)
    return sum


def printLayerShapes(pastKeyValues) -> None:
    print("\nKV Cache structure")
    for layerIndex, (k, v) in enumerate(pastKeyValues):
        print(f"Layer {layerIndex:02d} | K {tuple(k.shape)} | V {tuple(v.shape)}")
        
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained(modelName)
    model     = GPT2LMHeadModel.from_pretrained(modelName)
    model.to(device).eval()

    prompt = (
        "The future of artificial intelligence will likely "
        "transform every industry over the next decade."
    )

    inputIds       = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    generatedIds   = inputIds[0].cpu().tolist()
    pastKeyValues  = None
    allAttentions  = []                        

    csvPath = os.path.join(artifactDirectory, "cacheGrowth.csv")

    with open(csvPath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "sequenceLength", "cacheMb"])

        for step in range(maxTokens):
            with torch.no_grad():
                outputs = model(
                    input_ids=inputIds,
                    past_key_values=pastKeyValues,
                    use_cache=True,
                    output_attentions=True,
                    return_dict=True,
                )

            pastKeyValues = outputs.past_key_values
            if step == 0:
                printLayerShapes(pastKeyValues)

            totalCacheMb = kvCacheSizeMb(pastKeyValues)
            seqLen       = pastKeyValues[0][0].shape[2]

            print(f"\nStep {step + 1:02d}")
            print(f"Sequence Length : {seqLen}")
            print(f"KV Cache Size   : {totalCacheMb:.4f} MB")

            writer.writerow([step + 1, seqLen, totalCacheMb])

            allAttentions.append(
                tuple(a.cpu() for a in outputs.attentions)
            )
            
            nextToken = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generatedIds.append(nextToken.item())
            inputIds = nextToken
    for stepIdx, stepAttentions in enumerate(allAttentions):
        attnPath = os.path.join(attentionDirectory, f"attnStep{stepIdx:03d}.pt")
        torch.save(stepAttentions, attnPath)
    torch.cuda.empty_cache()

    print("\nfinal:\n")
    print(tokenizer.decode(generatedIds, skip_special_tokens=True))
    print(f"\nSaved:\n - {csvPath}\n - {attentionDirectory}/")
    
if __name__ == "__main__":
    main()