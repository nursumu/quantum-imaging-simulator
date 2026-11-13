"""Data-driven, cautious interpretation of comparison results.

A channel is called 'better' only when the difference in the mean metric exceeds
twice the combined standard deviation of the repeats; otherwise the result is
reported as 'not distinguishable'. Statements describe THIS simulation only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAIN = {"ch1": "Ch1 coherent", "ch2": "Ch2 covariance", "ch3": "Ch3 coincidence"}
EST = {"ch1": "coherent", "ch2": "cov", "ch3": "coincidence"}


def _main(df):
    return df[[EST.get(c) == e for c, e in zip(df["channel"], df["estimator"])]]


def compare_at(df, metric="SSIM") -> list[str]:
    out = []
    d = _main(df)
    for xv, g in d.groupby("x_value"):
        g = g.set_index("channel")
        if not {"ch1", "ch3"} <= set(g.index):
            continue
        r = {c: (g.loc[c, metric], g.loc[c, metric + "_std"]) for c in g.index}
        best = max(r, key=lambda c: r[c][0])
        others = [c for c in r if c != best]
        sig = all(r[best][0] - r[c][0] > 2 * np.hypot(r[best][1], r[c][1]) for c in others)
        out.append((xv, best, sig, r))
    return out


def summary_sentences(df: pd.DataFrame, kind: str, metric="SSIM") -> list[str]:
    xlab = {"photons": "N", "noise": "b", "thickness": "h", "time": "T"}.get(kind, "x")
    res = compare_at(df, metric)
    lines = []
    for xv, best, sig, r in res:
        vals = ", ".join(f"{MAIN[c]} {v:.3g}±{s:.2g}" for c, (v, s) in r.items())
        floor = {"SSIM": 0.02, "SNR_noise": 1.0, "CNR": 1.0}.get(metric, -np.inf)
        if r[best][0] < floor:
            lines.append(f"{xlab} = {xv:g}: all images are essentially noise ({metric} < {floor:g} for every channel); "
                         f"no meaningful ranking ({vals}).")
            continue
        if sig:
            lines.append(f"{xlab} = {xv:g}: **{MAIN[best]}** has the highest {metric} ({vals}).")
        else:
            lines.append(f"{xlab} = {xv:g}: no channel is distinguishably better in {metric} ({vals}).")
    floor = {"SSIM": 0.02, "SNR_noise": 1.0, "CNR": 1.0}.get(metric, -np.inf)
    res = [x for x in res if x[3][x[1]][0] >= floor]
    wins = [b for _, b, s, _ in res if s]
    if wins and all(w == "ch1" for w in wins):
        lines.append("Overall: coherent intensity imaging is best or equivalent at every tested point; no correlation-based advantage was found.")
    elif "ch3" in wins:
        pts = [f"{xv:g}" for xv, b, s, _ in res if s and b == "ch3"]
        lines.append(f"Overall: the coincidence image is best at {xlab} = {', '.join(pts)}. In this model, this advantage comes from "
                     "rejection of uncorrelated counts by coincidence timing, not from spatial entanglement (see the separable "
                     "control on the Validation page).")
    if "ch2" in wins:
        lines.append("The classical covariance image is best at some points; check the reference-variance condition and uncertainty.")
    return lines
