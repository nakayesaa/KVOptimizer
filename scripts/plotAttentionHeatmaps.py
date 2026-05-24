import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.attention_viz import (
    loadAllAttentionSteps,
    plotLayerHeatmaps,
    plotLayerSparsityCurve,
)

attentionDirectory = "artifacts/attentions"
artifactDirectory  = "artifacts"


def main():
    print("Attention Heatmaps")
    heatmapPath = os.path.join(artifactDirectory, "attentionHeatmaps.png")
    plotLayerHeatmaps(
        stepIndex=0,
        attentionDirectory=attentionDirectory,
        savePath=heatmapPath,
    )

    print("Layer Sparsity Analysis")
    allSteps    = loadAllAttentionSteps(attentionDirectory)
    sparsityPath = os.path.join(artifactDirectory, "layerSparsity.png")
    entropies    = plotLayerSparsityCurve(allSteps, savePath=sparsityPath)

    minIdx = entropies.index(min(entropies))
    maxIdx = entropies.index(max(entropies))
    print(f"Most sparse layer: L{minIdx:02d} (entropy {entropies[minIdx]:.4f})")
    print(f"Least sparse layer: L{maxIdx:02d} (entropy {entropies[maxIdx]:.4f})")
    print(f"\nSaved File:")
    print(f"{heatmapPath}")
    print(f"{sparsityPath}")

if __name__ == "__main__":
    main()
