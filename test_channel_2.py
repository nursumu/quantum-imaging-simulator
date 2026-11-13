"""Channel 2 validation suite."""
from __future__ import annotations

import numpy as np

from core.config import SimConfig
from models.cell_model import build_cell, uniform_object
from models.classical_correlation_model import ClassicalCorrelationModel
from calculations.metrics import detector_masks
from validation.common import check, assert_all


def run_checks(cfg: SimConfig | None = None, progress=None) -> list:
    cfg = (cfg or SimConfig()).with_(background=0.0, dark=0.0)
    rng = np.random.default_rng(21)
    out = []
    step = lambda f: progress(f) if progress else None
    hi = cfg.with_(photons=1e9)

    m1 = ClassicalCorrelationModel(hi, uniform_object(cfg, 1.0))
    fr = m1.frames(300, rng, keep_fields=True)
    E = np.fft.ifft2(m1.src.coeffs(rng, 200))
    I = np.abs(E) ** 2
    g2 = np.mean(I ** 2) / np.mean(I) ** 2
    out.append(check("Thermal source statistics", abs(g2 - 2) < 0.05 and abs(I.mean() - 1) < 0.02,
                     f"⟨I⟩={I.mean():.3f}, g²(0)={g2:.3f}", "⟨I⟩=1, g²(0)=2±0.05",
                     "Circular complex-Gaussian (thermal) field: photon bunching g²(0)=2.", "physics"))
    s = m1.sample(rng, M=1000)
    ex1 = np.max(abs(m1.sys["cov"] - 1))
    out.append(check("Zero sample absorption", ex1 < 0.02 and abs(np.nanmean(s["images"]["cov"]) - 1) < 0.02,
                     f"exact max|T−1|={ex1:.1e}; sampled ⟨T_cov⟩={np.nanmean(s['images']['cov']):.4f}", "|T−1|<0.02",
                     "t=1: Cov(I_s,I_r)=Var(I_r) → T_cov=1.", "boundary"))
    step(0.15)
    mT = ClassicalCorrelationModel(hi, uniform_object(cfg, 0.4, 0.7))
    sT = mT.sample(rng, M=1000)
    v = {k: float(np.nanmean(x)) for k, x in sT["images"].items()}
    out.append(check("Known sample transmission T=0.4", all(abs(x - 0.4) < 0.02 for x in v.values()),
                     ", ".join(f"{k}={x:.4f}" for k, x in v.items()), "all within 0.02", "", "physics"))
    step(0.3)
    cs, meas = [0.0, 0.25, 0.5, 0.75, 1.0], []
    for c in cs:
        mc = m1.variant(corr_strength=c)
        meas.append(float(np.nanmean(mc.sample(rng, N=1e8, M=400)["images"]["cov"])))
    slope = np.polyfit(cs, meas, 1)[0]
    out.append(check("Uncorrelated-beams control (c = 0)", abs(meas[0]) < 0.02, f"⟨T_cov⟩(c=0) = {meas[0]:+.4f}", "≈ 0",
                     "Independent beams: no cross-covariance, no correlation image.", "control"))
    out.append(check("Increasing correlation strength", abs(slope - 1) < 0.05,
                     ", ".join(f"c={c}: {x:.3f}" for c, x in zip(cs, meas)) + f"; slope={slope:.3f}", "slope 1±0.05",
                     "Gaussian moment theorem: Cov ∝ |⟨E E_r*⟩|² = c.", "physics"))
    step(0.5)
    cell = build_cell(cfg)
    mc = ClassicalCorrelationModel(cfg, cell)
    roi = detector_masks(cell, cfg.bin_factor)["cell"]
    bias, rmse = [], []
    for b in [0.0, 10.0, 100.0, 1000.0]:
        bb, rr = [], []
        for _ in range(3):
            o = mc.sample(rng, N=1e6, noise=b, M=100)
            d = o["images"]["cov"][roi] - mc.sys["cov"][roi]
            bb.append(np.nanmean(d))
            rr.append(np.sqrt(np.nanmean(d ** 2)))
        bias.append(np.mean(bb))
        rmse.append(np.mean(rr))
    out.append(check("Increasing background noise", max(map(abs, bias)) < 0.05 and rmse[-1] > rmse[0],
                     "bias=" + ",".join(f"{x:+.3f}" for x in bias) + "; RMSE=" + ",".join(f"{x:.3f}" for x in rmse),
                     "|bias|<0.05, RMSE grows", "Uncorrelated background cancels in the covariance mean, adds variance.", "noise"))
    step(0.7)
    Ms, errs = [10, 30, 100, 300, 1000], []
    for M in Ms:
        o = mc.sample(rng, N=1e10, M=M)
        errs.append(np.sqrt(np.nanmean((o["images"]["cov"][roi] - mc.sys["cov"][roi]) ** 2)))
    sl = np.polyfit(np.log(Ms), np.log(errs), 1)[0]
    out.append(check("Frames: convergence of the correlation estimate", -0.62 < sl < -0.38,
                     "RMSE=" + ",".join(f"{e:.3f}" for e in errs) + f"; slope={sl:.2f}", "M^-0.5 (±0.12)",
                     "Finite-frame estimate converges to the exact ensemble covariance.", "convergence"))
    o = mc.sample(rng, N=1e10, M=1000, keep=True)
    zres = (o["images"]["cov"] - mc.sys["cov"]) / o["se_cov_T"]
    out.append(check("Uncertainty estimator calibration", 0.8 < np.nanstd(zres) < 1.25 and o["extras"]["g2_sr_pooled"] <= 2,
                     f"std((T−T_exact)/SE)={np.nanstd(zres):.2f}; pooled g²_sr={o['extras']['g2_sr_pooled']:.3f}",
                     "0.8–1.25; g²_sr ≤ 2", "SE matches the actual error; classical Cauchy–Schwarz bound respected.", "statistics"))
    step(0.85)
    zs = []
    for mode in ("dose", "total"):
        o = mc.sample(rng, N=1e5, noise=1.0, M=100, mode=mode)["extras"]
        e = o["expected_signal_counts"] + cfg.n_det_pixels
        zs.append(abs(o["detected_signal"] - e) / np.sqrt(1.3 * e))
    tot = mc.sample(rng, N=1e5, M=100, mode="total")["extras"]
    out.append(check("Photon-budget consistency", max(zs) < 5 and tot["N_incident"] + tot["N_reference"] == 1e5,
                     f"max|z|={max(zs):.2f}; total mode N_s+N_r={tot['N_incident']+tot['N_reference']:.0f}",
                     "|z|<5; N_s+N_r=N", "", "budget"))
    step(1.0)
    return out


def test_channel_2():
    assert_all(run_checks())
