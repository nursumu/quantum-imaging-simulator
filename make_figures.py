#!/usr/bin/env python3
"""Static (Matplotlib) thesis figures from results/sweeps_reference.csv -> results/figures/*.png"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from core.experiment import MAIN_ESTIMATORS  # noqa: E402

COL = {"ch1": "#2a78d6", "ch2": "#eb6834", "ch3": "#e87ba4"}
MRK = {"ch1": "o", "ch2": "D", "ch3": "v"}
XL = {"photons": ("Photon budget N", True), "noise": ("Background b (counts/px)", True),
      "thickness": ("Cell thickness h (µm)", False), "time": ("Measurement time T (relative)", True)}


def main():
    df = pd.read_csv(ROOT / "results" / "sweeps_reference.csv")
    out = ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e6"})
    for kind, (xl, logx) in XL.items():
        d = df[df["tag"] == kind]
        if d.empty:
            continue
        mets = ["SNR_noise", "CNR", "SSIM"] + (["resolution_um"] if kind == "photons" else [])
        fig, axs = plt.subplots(1, len(mets), figsize=(4 * len(mets), 3.3))
        for ax, m in zip(axs, mets):
            for lab, ch, est in MAIN_ESTIMATORS:
                g = d[(d.channel == ch) & (d.estimator == est)].sort_values("x_value")
                ax.errorbar(g.x_value, g[m], yerr=g.get(m + "_std"), color=COL[ch], marker=MRK[ch], ms=6,
                            capsize=3, lw=2, label=lab)
            ax.set_xscale("log" if logx else "linear")
            if m == "SNR_noise":
                ax.set_yscale("log")
            ax.set_xlabel(xl)
            ax.set_title({"SNR_noise": "SNR", "resolution_um": "10–90 % edge width (µm)"}.get(m, m))
        axs[-1].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"quality_vs_{kind}.png", dpi=200)
        plt.close(fig)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
