import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    model.to(device)
    model.eval()

    print(f"Model device: {next(model.parameters()).device}")
    print("\nArchitecture")
    print(f"Layers: {model.config.n_layer}")
    print(f"Heads: {model.config.n_head}")
    print(f"Hidden: {model.config.n_embd}")

    prompt = "how to make a string in python"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False
        )

    print("\nGenerated text:")
    print(tokenizer.decode(output[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()

