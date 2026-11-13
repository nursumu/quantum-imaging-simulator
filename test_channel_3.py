"""Channel 3 validation suite."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from calculations.metrics import detector_masks, reference_transmittance, ssim
from core.config import SimConfig
from models.cell_model import build_cell, uniform_object
from models.entangled_photon_model import (EntangledPhotonModel, epr_analytic, measured_epr_product,
                                           sample_pairs_1d, _packet)
from validation.common import check, assert_all

SD, SS = 0.8, 400.0


def run_checks(cfg: SimConfig | None = None, progress=None) -> list:
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(31)
    out = []
    step = lambda f: progress(f) if progress else None

    xs, xi = sample_pairs_1d(rng, 200000, "entangled", SD, SS)
    xg = np.linspace(-40, 40, 801)
    A, B = np.meshgrid(xg, xg, indexing="ij")
    Psi = np.exp(-(A + B) ** 2 / (4 * 8.0 ** 2) - (A - B) ** 2 / (4 * SD ** 2))
    lam = np.linalg.svd(Psi, compute_uv=False) ** 2
    lam /= lam.sum()
    K_num, K_an = 1 / np.sum(lam ** 2), epr_analytic(SD, 8.0)["schmidt_per_axis"]
    P = 48
    g = _packet(P, 24, 24, 0.5, SD)
    cc = np.arange(P) * 0.5 - 12
    vpk = np.sum(g ** 2 * cc[None, :] ** 2)
    ok = (abs(np.var(xs - xi) / SD ** 2 - 1) < 0.02 and abs(np.var(xs + xi) / SS ** 2 - 1) < 0.02
          and abs(K_num / K_an - 1) < 0.02 and abs(vpk / SD ** 2 - 1) < 0.05)
    out.append(check("Known biphoton spatial correlations", ok,
                     f"Var(x_s−x_i)={np.var(xs-xi):.3f} (σ_D²={SD**2:.2f}); Schmidt K: SVD {K_num:.2f} vs analytic {K_an:.2f}; "
                     f"conditional packet var {vpk:.3f}", "within 2 % (packet 5 %)",
                     "Sampled |Ψ|² reproduces design widths; SVD of discretised Ψ confirms non-factorisable state.", "physics"))
    pe, ce = measured_epr_product(rng, 20000, "entangled", SD, SS)
    ps, cs = measured_epr_product(rng, 20000, "separable", SD, SS)
    out.append(check("EPR entanglement witness", ce[1] < 0.25 and cs[0] > 0.25,
                     f"entangled {pe:.1e} (CI ≤ {ce[1]:.1e}); separable {ps:.2f} (CI ≥ {cs[0]:.2f})",
                     "entangled < 1/4 ≤ separable", "Reid EPR criterion with pixelated near/far-field data.", "entanglement"))
    step(0.15)
    cell = build_cell(cfg)
    me = EntangledPhotonModel(cfg, cell, SD, SS)
    step(0.35)
    msep = EntangledPhotonModel(cfg, cell, SD, SS, state="separable", flat=None)
    ref = reference_transmittance(cell["tau"], cfg.bin_factor)
    d = float(np.sqrt(np.mean((me.sys["coincidence"] - msep.sys["coincidence"]) ** 2)))
    out.append(check("Separable-state control", d < 0.02,
                     f"RMS diff of system images {d:.4f}; SSIM vs T_ref {ssim(me.sys['coincidence'], ref):.3f} / "
                     f"{ssim(msep.sys['coincidence'], ref):.3f}", "<0.02",
                     "Same near-field coincidence image without entanglement → image quality is not entanglement-specific.", "control"))
    step(0.55)
    flat = (me.K_flat, me.Ks_flat)
    m1 = EntangledPhotonModel(cfg, uniform_object(cfg, 1.0), SD, SS, flat=flat)
    T1 = float(np.mean(m1.sample(rng, N=1e6, noise=0.0)["images"]["coincidence"]))
    rs = m1.Pc.sum(1)
    out.append(check("No sample absorption", abs(T1 - 1) < 0.01 and rs.min() > 0.99 and rs.max() <= 1 + 1e-6,
                     f"⟨T̂⟩={T1:.4f}; ΣP per pair ∈ [{rs.min():.4f}, {rs.max():.4f}]", "|T−1|<0.01; ΣP ≤ 1", "", "boundary"))
    mT = EntangledPhotonModel(cfg, uniform_object(cfg, 0.4, 0.5), SD, SS, flat=flat)
    T4 = float(np.mean(mT.sample(rng, N=1e6, noise=0.0)["images"]["coincidence"]))
    out.append(check("Known sample transmission T=0.4", abs(T4 - 0.4) < 0.01, f"⟨T̂⟩={T4:.4f}", "|T−0.4|<0.01",
                     "Coincidences scale with |t|²; constant phase unobservable.", "physics"))
    step(0.7)
    etas, co, si = [0.25, 0.5, 1.0], [], []
    for e in etas:
        o = m1.sample(rng, N=2e5, noise=0.0, eta_s=e, eta_i=e)["extras"]
        co.append(o["coincidences_true"])
        si.append(o["signal_detected"])
    sc, ss_ = np.polyfit(np.log(etas), np.log(co), 1)[0], np.polyfit(np.log(etas), np.log(si), 1)[0]
    out.append(check("Detector losses", abs(sc - 2) < 0.05 and abs(ss_ - 1) < 0.05,
                     f"log-slopes: coincidences {sc:.3f}, singles {ss_:.3f}", "2 and 1", "Not every pair is detected.", "physics"))
    res = []
    for mu in [0.1, 1.0, 10.0]:
        o = m1.sample(rng, N=1e5, noise=5.0, mu=mu, keep=True)
        ex, mp = o["extras"], o["maps"]
        pred = float(np.sum(mp["S_s"] * uniform_filter(mp["S_i"], 3, mode="constant") * 9) / ex["windows"])
        res.append((mu, ex["coincidences_accidental"], pred, ex["accidentals_delayed"], ex["CAR"]))
    ok = all(abs(a - p) < 0.1 * p + 3 * np.sqrt(p) and abs(dl - a) < 0.1 * a + 4 * np.sqrt(a + 1) for _, a, p, dl, _ in res)
    out.append(check("Accidental coincidences", ok and res[0][4] > res[-1][4],
                     "; ".join(f"μ={m}: acc {a:.0f}, pred {p:.0f}, delayed {dl:.0f}" for m, a, p, dl, _ in res),
                     "within 10 %; CAR falls with μ", "Accidentals ≈ Σ S_s S_i(W)/M; delayed-window estimate unbiased.", "noise"))
    step(0.85)
    roi = detector_masks(cell, cfg.bin_factor)["cell"]
    bl = []
    for b in [0.0, 10.0, 100.0]:
        bl.append((b, np.mean([np.mean(me.sample(rng, N=1e6, noise=b)["images"]["coincidence"][roi] - me.sys["coincidence"][roi])
                               for _ in range(2)])))
    out.append(check("Background noise", all(abs(x) < 0.02 for _, x in bl),
                     ", ".join(f"b={b:.0f}: bias {x:+.4f}" for b, x in bl), "|bias|<0.02", "", "noise"))
    o = me.sample(rng, N=1e5, noise=1.0)["extras"]
    bk = o["pairs_generated"] == o["signal_lost_sample_or_pupil"] + o["signal_off_sensor"] + o["signal_reaching_detector"]
    zt = abs(o["coincidences_true"] - o["expected_true"]) / np.sqrt(o["expected_true"])
    out.append(check("Photon-pair budget consistency", bk and zt < 5 and o["coincidences_true"] <= o["both_detected"],
                     f"generated {o['pairs_generated']}, det. signal {o['signal_detected']}, det. idler {o['idler_detected']}, "
                     f"true {o['coincidences_true']:.0f} (exp. {o['expected_true']:.0f}), accidental {o['coincidences_accidental']:.0f}",
                     "exact bookkeeping; |z|<5", "", "budget"))
    out.append(check("Cauchy–Schwarz (photon-number correlation)", o["g2_si"] ** 2 > o["g2_ss"] * o["g2_ii"],
                     f"g²_si={o['g2_si']:.1f}, g²_ss={o['g2_ss']:.3f}, g²_ii={o['g2_ii']:.3f}", "g²_si² > g²_ss g²_ii",
                     "Nonclassical time correlation (not a spatial-entanglement test).", "statistics"))
    step(1.0)
    return out


def test_channel_3():
    assert_all(run_checks())
