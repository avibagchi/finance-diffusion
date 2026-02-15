#!/usr/bin/env python3
"""
Recreate Factordiff-only heatmaps for k1, k170, k350 from saved weight CSVs.
Shows top 25 assets by mean weight; axis and title labels at font size 25.
Saves each heatmap as a PDF in sweep_results/k{K}/plots/.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_BASE = SCRIPT_DIR.parent / "sweep_results"
HP_SUFFIX = "_im0_sd0_h256_d6_e1000_lr0p0001"
FONT_SIZE = 25
TICK_FONT_SIZE = 12
MAX_ASSETS_HEATMAP = 25

def main():
    for k in [1, 170, 350]:
        csv_path = RESULTS_BASE / f"k{k}" / "plots" / f"weights_factordiff_im0_sd0_k{k}_h256_d6_e1000_lr0p0001.csv"
        if not csv_path.exists():
            # try glob in case filename differs
            plots_dir = RESULTS_BASE / f"k{k}" / "plots"
            candidates = list(plots_dir.glob("weights_factordiff*.csv"))
            if not candidates:
                print(f"Skip k{k}: no factordiff CSV in {plots_dir}")
                continue
            csv_path = candidates[0]

        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        months = data[:, 0]
        w = data[:, 1:]  # (n_test, D)
        n_test, D = w.shape

        mean_w = np.mean(w, axis=0)
        max_n = min(MAX_ASSETS_HEATMAP, D)
        top_idx = np.argsort(mean_w)[::-1][:max_n]
        w_top = w[:, top_idx].T  # (max_n, n_test) for imshow

        vmax = np.percentile(w * 100, 99)
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(w_top * 100, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_xlabel("Month", fontsize=FONT_SIZE)
        ax.set_ylabel("Asset (top by mean weight)", fontsize=FONT_SIZE)
        ax.set_title(f"Asset Weights (Factors={k})", fontsize=FONT_SIZE)
        ax.set_yticks(np.arange(max_n))
        ax.set_yticklabels([str(i) for i in range(max_n)], fontsize=TICK_FONT_SIZE)
        ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
        cbar = plt.colorbar(im, ax=ax, label="Weight (%)")
        cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
        cbar.set_label("Weight (%)", fontsize=FONT_SIZE)
        fig.tight_layout()
        out_path = RESULTS_BASE / f"k{k}" / "plots" / f"factordiff_heatmap_k{k}_fs25.pdf"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
