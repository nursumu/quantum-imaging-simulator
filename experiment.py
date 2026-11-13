"""Channel-independent experiment runner used by the comparison page and scripts.

Fairness rules enforced here:
  * all channels receive the same SimConfig (same cell, optics, eta, background flux);
  * photon budget split by core.config.photon_split ('dose' or 'total');
  * background/dark counts are counts per pixel over the SAME acquisition time;
    Ch2 spreads them over its frames, Ch3 over its coincidence windows;
  * measurement-time sweeps scale photons AND background linearly with time T,
    with the same source flux for every channel.
"""
from __future__ import annotations

import zlib

import numpy as np

from calculations.metrics import (detector_masks, reference_transmittance, image_metrics, edge_resolution,
                                  summarize)
from core.config import SimConfig
from models.cell_model import build_cell, edge_target
from models.classical_correlation_model import ClassicalCorrelationModel
from models.coherent_model import CoherentModel
from models.entangled_photon_model import EntangledPhotonModel

ESTIMATORS = [  # (label, channel, estimator) — fixed order => fixed colours
    ("Ch1 · coherent intensity", "ch1", "coherent"),
    ("Ch2 · speckle intensity (signal arm)", "ch2", "conv"),
    ("Ch2 · reference-normalised (1st order)", "ch2", "ratio"),
    ("Ch2 · intensity covariance C_sr", "ch2", "cov"),
    ("Ch3 · entangled-pair coincidences", "ch3", "coincidence"),
]
MAIN_ESTIMATORS = [ESTIMATORS[0], ESTIMATORS[3], ESTIMATORS[4]]


def seed_for(*parts) -> int:
    return zlib.crc32(repr(parts).encode()) & 0x7FFFFFFF


def make_model(channel, cfg, obj, ch2_opts=None, ch3_opts=None, flat=None):
    if channel == "ch1":
        return CoherentModel(cfg, obj)
    if channel == "ch2":
        return ClassicalCorrelationModel(cfg, obj, **(ch2_opts or {}))
    return EntangledPhotonModel(cfg, obj, flat=flat, **(ch3_opts or {}))


class ModelSet:
    """Models of the three channels for one configuration (cell + edge target)."""

    def __init__(self, cfg: SimConfig, ch2_opts=None, ch3_opts=None, channels=("ch1", "ch2", "ch3")):
        self.cfg = cfg
        self.cell = build_cell(cfg)
        self.edge = edge_target(cfg)
        self.ref = reference_transmittance(self.cell["tau"], cfg.bin_factor)
        self.masks = detector_masks(self.cell, cfg.bin_factor)
        self.ch2_opts, self.ch3_opts = ch2_opts or {}, ch3_opts or {}
        self.models, self.edge_models = {}, {}
        for ch in channels:
            m = make_model(ch, cfg, self.cell, self.ch2_opts, self.ch3_opts)
            self.models[ch] = m
            flat = (m.K_flat, m.Ks_flat) if ch == "ch3" else None
            self.edge_models[ch] = make_model(ch, cfg, self.edge, self.ch2_opts, self.ch3_opts, flat=flat)


def _sample(model, channel, rng, N, noise, bg_factor, mode, time_factor):
    if channel == "ch2":
        M = max(2, int(round(model.M * time_factor)))
        return model.sample(rng, N=N, noise=noise, bg_factor=bg_factor, mode=mode, M=M)
    return model.sample(rng, N=N, noise=noise, bg_factor=bg_factor, mode=mode)


def evaluate(ms: ModelSet, N, noise, bg_factor=1.0, mode=None, repeats=3, tag="", resolution=True,
             time_factor=1.0, estimators=ESTIMATORS):
    cfg = ms.cfg
    mode = mode or cfg.budget_mode
    rows = []
    for ch in ms.models:
        per, ex, res = {}, [], {}
        for r in range(repeats):
            rng = np.random.default_rng(seed_for(cfg.seed, ch, tag, N, noise, bg_factor, mode, time_factor, r))
            out = _sample(ms.models[ch], ch, rng, N, noise, bg_factor, mode, time_factor)
            ex.append({k: v for k, v in out["extras"].items() if np.isscalar(v)})
            for name, img in out["images"].items():
                per.setdefault(name, []).append(image_metrics(img, ms.ref, ms.models[ch].sys[name], ms.masks))
            if resolution:
                rng = np.random.default_rng(seed_for(cfg.seed, ch, "edge", tag, N, noise, mode, time_factor, r))
                eo = _sample(ms.edge_models[ch], ch, rng, N, noise, bg_factor, mode, time_factor)
                for name, img in eo["images"].items():
                    res.setdefault(name, []).append(edge_resolution(img, cfg.dx_det, 0.0, 8.0,
                                                                    1 - ms.edge["T_dark"])["width_10_90_um"])
        exs = summarize(ex)
        for label, c, est in estimators:
            if c != ch or est not in per:
                continue
            row = dict(label=label, channel=ch, estimator=est, N=N, noise=noise, bg_factor=bg_factor, mode=mode,
                       thickness_um=cfg.thickness_um, time_factor=time_factor, tag=tag, repeats=repeats)
            row.update(summarize(per[est]))
            if res:
                rr = np.array(res[est], float)
                row["resolution_um"] = float(np.nanmean(rr)) if np.isfinite(rr).any() else np.nan
                row["resolution_um_std"] = float(np.nanstd(rr)) if np.isfinite(rr).sum() > 1 else np.nan
            row.update({"x_" + k: v for k, v in exs.items()})
            rows.append(row)
    return rows


def sweep(cfg, kind, values, ch2_opts=None, ch3_opts=None, repeats=2, progress=None, base_ms=None):
    """kind in {'photons','noise','thickness','time'}; returns result rows."""
    rows = []
    ms = base_ms
    for i, v in enumerate(values):
        n0 = len(rows)
        if kind == "thickness":
            ms = ModelSet(cfg.with_(thickness_um=v), ch2_opts, ch3_opts)
            rows += evaluate(ms, cfg.photons, cfg.noise_counts, repeats=repeats, tag="thickness", resolution=False)
        else:
            if ms is None:
                ms = ModelSet(cfg, ch2_opts, ch3_opts)
            if kind == "photons":
                rows += evaluate(ms, v, cfg.noise_counts, repeats=repeats, tag="photons", resolution=True)
            elif kind == "noise":
                rows += evaluate(ms, cfg.photons, v, repeats=repeats, tag="noise", resolution=False)
            elif kind == "time":
                rows += evaluate(ms, cfg.photons * v, cfg.noise_counts * v, repeats=repeats, tag="time",
                                 resolution=False, time_factor=v)
        for r in rows[n0:]:
            r["x_value"] = v
        if progress:
            progress((i + 1) / len(values), f"{kind} = {v:g}")
    return rows
