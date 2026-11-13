#!/usr/bin/env python3
"""Generate the precomputed reference comparison results in results/.

Usage:  python scripts/run_sweeps.py [--repeats 3] [--quick]
Writes results/sweeps_reference.csv and results/sweeps_reference_meta.json.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.config import SimConfig  # noqa: E402
from core.experiment import ModelSet, sweep  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cfg = SimConfig()
    ch2, ch3 = dict(frames=100, speckle_um=2.5), dict(sigma_D=0.8, mu=0.1)
    t0 = time.time()
    log = lambda f, m: print(f"  [{time.time()-t0:6.0f}s] {m}", flush=True)
    ms = ModelSet(cfg, ch2, ch3)
    plan = {"photons": [1e2, 1e3, 1e4, 1e5] + ([] if a.quick else [1e6]),
            "noise": [0.1, 1.0, 10.0, 100.0, 1000.0],
            "thickness": [2.0, 5.0, 10.0],
            "time": [0.01, 0.1, 1.0, 10.0]}
    rows = []
    for k, vals in plan.items():
        print(f"sweep {k}")
        rows += sweep(cfg, k, vals, ch2, ch3, repeats=a.repeats, progress=log, base_ms=None if k == "thickness" else ms)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "sweeps_reference.csv", index=False)
    meta = dict(config=cfg.as_dict(), ch2=ch2, ch3=ch3, repeats=a.repeats, runtime_s=round(time.time() - t0),
                numpy=np.__version__, python=platform.python_version())
    (out / "sweeps_reference_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"done in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
