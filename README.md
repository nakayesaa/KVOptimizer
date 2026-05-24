# Simple KV Cache Optimization

A project focused on implementing and optimizing KV cache eviction algorithms (StreamingLLM, Heavy Hitter Oracle, and Hybrid mechanisms) to enable efficient long-context generation under constrained hardware(fahhh)

## Getting Started

### Prerequisites
* Python 3.10+
* PyTorch (with CUDA support)
* Transformers
* Datasets (for WikiText perplexity evaluations)
* Matplotlib & Seaborn (for visualization)

### Installation
```bash
pip install torch transformers datasets matplotlib seaborn
```

## Running Benchmarks

To run the unified driver script:
```bash
python run_benchmark.py --cache_type full --budget 128
```
