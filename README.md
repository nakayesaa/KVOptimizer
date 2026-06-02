# KV Cache Optimization

A lightweight research implementation of KV cache eviction strategies for efficient autoregressive generation under limited GPU memory.

This project compares four cache configurations on `gpt2-medium`:

* **Full Cache**: keeps every past key-value state as the quality baseline.
* **StreamingLLM**: keeps attention sink tokens and a recent sliding window.
* **H2O**: keeps tokens with the highest cumulative attention scores.
* **Hybrid Cache**: combines attention sinks, recent context, and H2O heavy hitters.

## Why This Project?

During token-by-token text generation, the KV cache grows with sequence length. This improves context retention, but increases memory usage over time.

The goal of this project is to evaluate whether bounded KV-cache strategies can reduce cache growth while preserving model quality.

## Project Structure

```text
KVOptimizer/
├── analysis/
│   └── attention_viz.py
├── benchmark/
│   ├── memory.py
│   └── perplexity.py
├── cache/
│   ├── base.py
│   ├── full.py
│   ├── streaming.py
│   ├── h2o.py
│   └── hybrid.py
├── scripts/
│   ├── inspectKVCache.py
│   ├── plotAttentionHeatmaps.py
│   ├── runBaseline.py
│   ├── generateWithCache.py
│   ├── inspectManagedKVCache.py
│   └── runPolicyBenchmark.py
├── tests/
│   └── testCachePolicies.py
└── artifacts/
```

## Installation

### Requirements

* Python 3.10+
* PyTorch with CUDA support recommended
* Hugging Face Transformers
* Datasets
* Matplotlib
* Seaborn

### Install Dependencies

```bash
pip install torch transformers datasets matplotlib seaborn
```

## Phase 1: Baseline and Attention Analysis

Inspect native KV-cache growth and save attention tensors:

```bash
python scripts/inspectKVCache.py
```

Run the full-cache baseline benchmark:

```bash
python scripts/runBaseline.py
```

Generate attention heatmaps and layer sparsity visualization:

```bash
python scripts/plotAttentionHeatmaps.py
```

## Phase 2 and Phase 3: Custom KV Cache Policies

Run the cache-policy smoke tests:

```bash
python tests/testCachePolicies.py
```

Inspect bounded StreamingLLM cache growth:

```bash
python scripts/inspectManagedKVCache.py --cacheType streaming --budget 128 --sinkTokens 4 --maxTokens 200
```

Generate text using the Hybrid cache:

```bash
python scripts/generateWithCache.py --cacheType hybrid --budget 128 --sinkTokens 4 --recentTokens 32 --maxTokens 200
```

## Benchmark All Policies

Run Full Cache, StreamingLLM, H2O, and Hybrid under the same cache budget:

```bash
python scripts/runPolicyBenchmark.py --cacheType all --budget 128 --sinkTokens 4 --recentTokens 32 --genSteps 200 --pplSeqLen 512 --pplWindows 10
```

Results are saved to:

```text
artifacts/policyBenchmarkResults.json
```

## Initial Benchmark Result

Configuration:

```text
Model          : gpt2-medium
Cache budget   : 128 tokens
Sink tokens    : 4
Hybrid recent  : 32
Generation     : 200 tokens
PPL windows    : 10
```

| Cache Policy | Perplexity | Peak VRAM (MB) | Tokens/sec | Final KV Length |
| ------------ | ---------: | -------------: | ---------: | --------------: |
| Full Cache   |    29.9570 |        1462.75 |      14.37 |             205 |
| StreamingLLM |   204.4938 |        1434.06 |      26.99 |             128 |
| H2O          |   105.5638 |        1434.26 |      22.86 |             128 |
| Hybrid Cache |    35.9982 |        1434.26 |      33.32 |             128 |

## Key Finding

At a cache budget of `128` tokens, the **Hybrid Cache** produced the strongest quality-efficiency trade-off.

* Full Cache achieved the best perplexity, but its KV cache continued growing.
* StreamingLLM successfully bounded the cache, but quality degraded significantly.
* H2O retained useful historical tokens better than pure sliding-window eviction.
* Hybrid kept the cache bounded while achieving perplexity close to Full Cache.

In this initial experiment, Hybrid reduced the final KV-cache length from `205` to `128` tokens while increasing perplexity only from `29.96` to `36.00`.

## Method Summary

### Full Cache

Stores all past key-value states and acts as the baseline.

### StreamingLLM

Retains:

```text
attention sink tokens + most recent tokens
```

This bounds cache growth but may remove older context that remains important.

### H2O

Retains:

```text
attention sink tokens + highest cumulative-attention tokens + newest token
```

This prioritizes tokens that consistently receive attention during generation.

### Hybrid Cache

Retains:

```text
attention sink tokens + recent context window + heavy-hitter tokens
```

This combines local continuity with long-range important-token retention.

## Outputs

Generated experiment artifacts are stored in:

```text
artifacts/
├── baselineResults.json
├── policyBenchmarkResults.json
├── cacheGrowth.csv
├── streamingCacheGrowth.csv
├── h2oCacheGrowth.csv
├── hybridCacheGrowth.csv
├── attentionHeatmaps.png
└── layerSparsity.png
```

## Limitation

This project uses `gpt2-medium`, which relies on absolute positional embeddings and has a limited maximum context length. Therefore, this implementation should be treated as a controlled KV-cache optimization experiment rather than an infinite-context deployment system.
