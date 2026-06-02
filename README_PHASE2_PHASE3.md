# Phase 2 and Phase 3 Implementation Overlay

This folder contains exact replacement and new files for `nakayesaa/KVOptimizer`.

## Copy into the repo

From the repository root, copy the folders from this overlay:

```bash
cp -r cache/* /path/to/KVOptimizer/cache/
cp -r benchmark/* /path/to/KVOptimizer/benchmark/
cp -r scripts/* /path/to/KVOptimizer/scripts/
mkdir -p /path/to/KVOptimizer/tests
cp -r tests/* /path/to/KVOptimizer/tests/
```

Existing `scripts/runBaseline.py`, `scripts/inspectKVCache.py`, and visualization scripts remain usable.

## Why the existing cache contract changed

GPT-2 returns `outputs.past_key_values` containing all K/V states already cached plus the newly processed tokens. The original `update(keyState, valueState, layerIndex)` interface appended the returned cache again, which would duplicate history once connected to inference.

The new interface is:

```python
pastKeyValues = cacheManager.manage(
    outputs.past_key_values,
    outputs.attentions if cacheManager.requiresAttentions else None,
)
```

It prunes the native returned cache and feeds the pruned result into the next forward pass.

## Policies

- `FullKVCache`: no eviction control baseline.
- `StreamingKVCache`: preserves initial sink tokens and most recent sliding window.
- `H2OKVCache`: preserves sinks, newest token, and top cumulative-attention heavy hitters.
- `HybridKVCache`: preserves sinks, a configurable recent window, and fills remaining capacity with heavy hitters.

## Commands

Run the pure policy smoke test:

```bash
python tests/testCachePolicies.py
```

Confirm the existing baseline still runs:

```bash
python scripts/runBaseline.py
```

Inspect bounded StreamingLLM cache growth:

```bash
python scripts/inspectManagedKVCache.py \
  --cacheType streaming \
  --budget 32 \
  --sinkTokens 4 \
  --maxTokens 100
```

Generate text under Hybrid cache:

```bash
python scripts/generateWithCache.py \
  --cacheType hybrid \
  --budget 128 \
  --sinkTokens 4 \
  --recentTokens 32 \
  --maxTokens 200
```

Run all policy benchmarks:

```bash
python scripts/runPolicyBenchmark.py \
  --cacheType all \
  --budget 128 \
  --sinkTokens 4 \
  --recentTokens 32 \
  --genSteps 200 \
  --pplSeqLen 512 \
  --pplWindows 10
```

Results are written to:

```text
artifacts/policyBenchmarkResults.json
```

## Evaluation note

`evaluatePerplexity()` is retained for the Phase 1 full-context baseline.

`evaluatePolicyPerplexity()` is intentionally token-by-token because each next-token prediction must observe the cache after the selected eviction policy has been applied. This is slower, but it measures the effect of StreamingLLM, H2O, and Hybrid eviction rather than full-context inference.

## GPT-2 position handling

GPT-2 uses absolute position embeddings and has a fixed position limit. After tokens are evicted, cache length is no longer the absolute token position. The new inference loops explicitly pass absolute `position_ids` so retained cached states and new tokens do not reuse incorrect positions.
