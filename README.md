# KV Cache Optimization

A project focused on implementing and optimizing KV cache eviction algorithms to enable efficient long-context generation under constrained hardware

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

---

## Phase 1: Foundations & Baselines

You can run the Phase 1 scripts to establish control metrics and visualize attention sparsity:
1. **Inspect Cache Growth & Save Attentions**
   ```bash
   python scripts/inspectKVCache.py
   ```
2. **Run Baseline Benchmarks (Memory & Perplexity)**
   ```bash
   python scripts/runBaseline.py
   ```
3. **Plot Attention Heatmaps & Sparsity Curve**
   ```bash
   python scripts/plotAttentionHeatmaps.py
   ```
---

## Running Phase 2 Benchmarks
Once custom caches are implemented, use the unified driver script:
```bash
python run_benchmark.py --cache_type full --budget 128
```
