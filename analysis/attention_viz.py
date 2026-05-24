import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
def loadAttentionStep(
    stepIndex: int,
    attentionDirectory: str = "artifacts/attentions",
) -> Tuple[torch.Tensor, ...]:
    path = os.path.join(attentionDirectory, f"attnStep{stepIndex:03d}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No attention file found at: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def loadAllAttentionSteps(
    attentionDirectory: str = "artifacts/attentions",
) -> List[Tuple[torch.Tensor, ...]]:
    files = sorted([
        f for f in os.listdir(attentionDirectory)
        if f.startswith("attnStep") and f.endswith(".pt")
    ])
    print(f"Found {len(files)} attention step files in '{attentionDirectory}'")
    return [
        torch.load(os.path.join(attentionDirectory, f), map_location="cpu", weights_only=False)
        for f in files
    ]

def attentionEntropy(attnWeights: torch.Tensor) -> float:
    eps     = 1e-9
    weights = attnWeights.float().clamp(min=eps)
    entropy = -(weights * weights.log()).sum(dim=-1)
    return entropy.mean().item()

def computeLayerSparsity(
    allSteps: List[Tuple[torch.Tensor, ...]],
) -> List[float]:
    if not allSteps:
        raise ValueError("No attention steps provided.")

    numLayers      = len(allSteps[0])
    layerEntropies = [[] for _ in range(numLayers)]

    for stepAttentions in allSteps:
        for layerIdx, attn in enumerate(stepAttentions):
            layerEntropies[layerIdx].append(attentionEntropy(attn))

    return [float(np.mean(e)) for e in layerEntropies]

def plotLayerHeatmaps(
    stepIndex: int = 0,
    attentionDirectory: str = "artifacts/attentions",
    layersToPlot: Optional[List[int]] = None,
    savePath: str = "artifacts/attentionHeatmaps.png",
) -> None:
    stepAttentions = loadAttentionStep(stepIndex, attentionDirectory)
    numLayers      = len(stepAttentions)

    if layersToPlot is None:
        step = max(1, numLayers // 6)
        layersToPlot = list(range(0, numLayers, step))[:6]

    numPlots = len(layersToPlot)
    cols     = math.ceil(numPlots / 2)

    fig, axes = plt.subplots(
        2, cols,
        figsize=(5 * cols, 10),
        facecolor="#0f1117",
    )
    axes = axes.flatten()

    fig.suptitle(
        f"Attention Heatmaps — Decode Step {stepIndex} (avg across heads)",
        fontsize=16, color="white", fontweight="bold", y=1.01,
    )

    for plotIdx, layerIdx in enumerate(layersToPlot):
        ax      = axes[plotIdx]
        attn    = stepAttentions[layerIdx] 
        avgAttn = attn[0].mean(dim=0).numpy()

        sns.heatmap(
            avgAttn, ax=ax, cmap="magma",
            vmin=0, vmax=avgAttn.max(),
            square=True, cbar=True, linewidths=0,
        )
        ax.set_title(f"Layer {layerIdx:02d}", color="white", fontsize=12, pad=8)
        ax.set_xlabel("Key position",   color="#aaaaaa", fontsize=9)
        ax.set_ylabel("Query position", color="#aaaaaa", fontsize=9)
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.set_facecolor("#0f1117")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")
    for i in range(numPlots, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(savePath, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"Saved heatmaps → {savePath}")

def plotLayerSparsityCurve(
    allSteps: List[Tuple[torch.Tensor, ...]],
    savePath: str = "artifacts/layerSparsity.png",
) -> List[float]:
    entropies = computeLayerSparsity(allSteps)
    numLayers = len(entropies)
    layers    = list(range(numLayers))

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")

    colors = plt.cm.plasma(np.linspace(0.2, 0.9, numLayers))
    ax.bar(layers, entropies, color=colors, edgecolor="#0f1117", linewidth=0.5)

    zCoeffs   = np.polyfit(layers, entropies, 1)
    trendLine = np.poly1d(zCoeffs)
    ax.plot(layers, trendLine(layers), color="#00ffcc", linewidth=2,
        linestyle="--", label="Trend", alpha=0.85)

    ax.set_title(
        "Attention Entropy per Layer  (lower = sparser = better eviction candidate)",
        color="white", fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Layer Index", color="#aaaaaa", fontsize=11)
    ax.set_ylabel("Mean Attention Entropy", color="#aaaaaa", fontsize=11)
    ax.tick_params(colors="#aaaaaa")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333344", labelcolor="white")

    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")

    minIdx = int(np.argmin(entropies))
    maxIdx = int(np.argmax(entropies))
    ax.annotate(
        f"Most sparse\nL{minIdx}", xy=(minIdx, entropies[minIdx]),
        xytext=(minIdx + 2, entropies[minIdx] + 0.05),
        color="#00ffcc", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#00ffcc"),
    )
    ax.annotate(
        f"Least sparse\nL{maxIdx}", xy=(maxIdx, entropies[maxIdx]),
        xytext=(maxIdx - 4, entropies[maxIdx] - 0.05),
        color="#ff6b6b", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#ff6b6b"),
    )

    plt.tight_layout()
    plt.savefig(savePath, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"Saved sparsity curve → {savePath}")
    
    print(f"\n{'Layer':<8} {'Mean Entropy':>14}")
    print("-" * 24)
    for i, e in enumerate(entropies):
        tag = " ← most sparse" if i == minIdx else (" ← least sparse" if i == maxIdx else "")
        print(f"  L{i:02d}   {e:>12.4f}{tag}")

    return entropies
