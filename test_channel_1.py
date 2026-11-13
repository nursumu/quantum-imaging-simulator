"""Channel 1 validation suite (run with pytest or from the app's Validation page)."""
from __future__ import annotations

import numpy as np

from calculations.photon_statistics import dispersion_test
from calculations.propagation import propagate, asm_transfer, bin_mean
from core.config import SimConfig
from models.cell_model import build_cell, uniform_object, smooth_random_field
from models.coherent_model import CoherentModel
from validation.common import check, assert_all


def run_checks(cfg: SimConfig | None = None, progress=None) -> list:
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(11)
    out = []
    lm, n, dx = cfg.wavelength_medium, cfg.grid, cfg.dx
    step = lambda f: progress(f) if progress else None

    c0 = build_cell(cfg.with_(mu_cyto=0, mu_nuc=0, mu_mem=0, mu_gran=0))
    dev = np.max(np.abs(np.abs(c0["t"]) ** 2 - 1))
    Uz = propagate(c0["t"], dx, lm, 10.0, band_limit=False)
    econ = abs(np.sum(np.abs(Uz) ** 2) / np.sum(np.abs(c0["t"]) ** 2) - 1)
    out.append(check("Zero absorption", dev < 1e-12 and econ < 1e-10, f"max||t|²−1|={dev:.1e}; ΔP after 10 µm={econ:.1e}",
                     "<1e-12, <1e-10", "μ_a = 0 → |t|² = 1; without pupil, ASM conserves power (Parseval).", "boundary"))
    step(0.1)
    mu, Ls, Tm = 0.1, np.array([0, 1, 2, 5, 10, 20.0]), []
    for L in Ls:
        m = CoherentModel(cfg, uniform_object(cfg, np.exp(-mu * L)))
        Tm.append(m.expected(1e6, 0.0)["mu_sig"].sum() / (cfg.qe * 1e6))
    Tm = np.array(Tm)
    err = np.max(np.abs(Tm - np.exp(-mu * Ls)) / np.exp(-mu * Ls))
    out.append(check("Beer–Lambert transmission", err < 1e-10, f"max rel. err={err:.1e}", "<1e-10",
                     "Uniform slab: detected fraction = exp(−μL) with optical depth τ = μL.", "physics"))
    phi = 3 * smooth_random_field(n, dx, 2.0, rng)
    t = np.exp(1j * phi)
    I0 = np.abs(propagate(t, dx, lm, 0.0)) ** 2
    Iz = np.abs(propagate(t, dx, lm, 15.0, band_limit=False)) ** 2
    out.append(check("Phase-only sample", np.max(abs(I0 - 1)) < 1e-12 and abs(Iz.mean() - 1) < 1e-10 and Iz.std() > 1e-3,
                     f"z=0: max|I−1|={np.max(abs(I0-1)):.1e}; z=15 µm: |⟨I⟩−1|={abs(Iz.mean()-1):.1e}, std={Iz.std():.3f}",
                     "uniform at z=0; power conserved; contrast at z>0",
                     "Phase is invisible to intensity at z = 0 and becomes Fresnel contrast after propagation.", "physics"))
    step(0.25)
    U = c0["t"] * 0.7
    out.append(check("Zero-distance propagation", np.all(asm_transfer(n, dx, lm, 0.0) == 1) and
                     np.max(abs(propagate(U, dx, lm, 0.0) - U)) < 1e-12, "H(z=0) ≡ 1; U_z = U_0", "exact",
                     "", "boundary"))
    z = 25.0
    pw = propagate(np.ones((n, n), complex), dx, lm, z, band_limit=False)
    e1 = np.max(abs(pw - np.exp(1j * 2 * np.pi * z / lm)))
    X = (np.arange(n) * dx)[None, :] * np.ones((n, 1))
    fx = 10 / (n * dx)
    tilt = np.exp(1j * 2 * np.pi * fx * X)
    kz = np.sqrt((2 * np.pi / lm) ** 2 - (2 * np.pi * fx) ** 2)
    e2 = np.max(abs(propagate(tilt, dx, lm, z, band_limit=False) - tilt * np.exp(1j * kz * z)))
    out.append(check("Uniform & tilted plane-wave propagation", e1 < 1e-10 and e2 < 1e-10, f"errors {e1:.1e}, {e2:.1e}",
                     "<1e-10", "Plane waves are eigenfunctions of free space: phase advances by k_z z.", "physics"))
    nf, dxf = 128, 0.2
    Xf = (np.arange(nf) * dxf)[None, :] * np.ones((nf, 1))
    f_ev = 50 / (nf * dxf)
    res = []
    for zz in (0.2, 0.5):
        a = np.abs(propagate(np.exp(1j * 2 * np.pi * f_ev * Xf), dxf, lm, zz, band_limit=False)).mean()
        res.append((a, np.exp(-2 * np.pi * zz * np.sqrt(f_ev ** 2 - 1 / lm ** 2))))
    ev = max(abs(a - b) / b for a, b in res)
    out.append(check("Evanescent decay", ev < 1e-10, f"|U| at z=0.2/0.5 µm: {res[0][0]:.4f}/{res[1][0]:.4f} (analytic "
                     f"{res[0][1]:.4f}/{res[1][1]:.4f})", "rel. err<1e-10",
                     f"f = {f_ev:.2f} c/µm > 1/λ_m = {1/lm:.2f}: decays as exp(−2πz√(f²−1/λ_m²)).", "physics"))
    step(0.45)
    ng, dxg, w0, zg = 256, 0.25, 3.0, 40.0
    cc = (np.arange(ng) - ng / 2) * dxg
    XX, YY = np.meshgrid(cc, cc)
    widths = []
    for (nn, dd) in [(128, 0.5), (256, 0.25)]:
        c2 = (np.arange(nn) - nn / 2) * dd
        X2, Y2 = np.meshgrid(c2, c2)
        Gz = np.abs(propagate(np.exp(-(X2 ** 2 + Y2 ** 2) / w0 ** 2), dd, lm, zg, band_limit=False)) ** 2
        widths.append(2 * np.sqrt(np.sum(X2 ** 2 * Gz) / Gz.sum()))
    w_an = w0 * np.sqrt(1 + (zg / (np.pi * w0 ** 2 / lm)) ** 2)
    out.append(check("Gaussian beam vs analytic + grid convergence", abs(widths[1] / w_an - 1) < 0.01 and
                     abs(widths[0] / widths[1] - 1) < 0.005,
                     f"w(40 µm): Δx=0.5 → {widths[0]:.4f}, Δx=0.25 → {widths[1]:.4f}, analytic {w_an:.4f} µm",
                     "<1 % vs analytic; <0.5 % between grids", "Independent check of H and numerical convergence.", "convergence"))
    cA = cfg.with_(grid=128, fov_um=64.0)
    cB = cfg.with_(grid=256, fov_um=64.0, bin_factor=4)
    IA = CoherentModel(cA, build_cell(cA)).I_det
    IB = CoherentModel(cB, build_cell(cB)).I_det
    conv_err = float(np.sqrt(np.mean((IA - IB) ** 2)))
    out.append(check("Cell image convergence (Δx 0.5 → 0.25 µm)", conv_err < 0.03,
                     f"RMS difference of detector images = {conv_err:.4f}", "<0.03",
                     "Same 1 µm detector pixels, optical sampling refined ×2 (cell texture is resampled, so a small "
                     "residual is expected).", "convergence"))
    step(0.65)
    cell = build_cell(cfg)
    m = CoherentModel(cfg, cell)
    zs, ok = [], True
    for N in [1e2, 1e3, 1e4, 1e5, 1e6]:
        e = m.expected(N, cfg.noise_counts)
        ok &= abs(e["mu_total"].sum() - (cfg.qe * N * m.I_det.mean() + cfg.noise_counts * cfg.n_det_pixels)) < 1e-6 * e["mu_total"].sum()
        tot = [rng.poisson(e["mu_total"]).sum() for _ in range(20)]
        zs.append(abs(np.mean(tot) - e["mu_total"].sum()) / np.sqrt(e["mu_total"].sum() / 20))
    out.append(check("Photon-budget consistency", ok and max(zs) < 4.5 and m.I_det.mean() < 1,
                     f"detected/incident = η×{m.I_det.mean():.4f}; max|z| of totals = {max(zs):.2f}",
                     "identity exact, |z|<4.5, losses kept", "No renormalisation removes absorption/pupil losses.", "budget"))
    mA = CoherentModel(cfg.with_(A0=3.7), cell)
    dA = np.max(abs(mA.expected(1e5, 0)["mu_sig"] - m.expected(1e5, 0)["mu_sig"]))
    out.append(check("Amplitude A₀ independence", dA < 1e-9, f"max Δμ = {dA:.1e}", "<1e-9",
                     "μ_sig normalised by Σ|U_in|² → independent of A₀.", "boundary"))
    e0 = m.expected(0.0, 0.0)["mu_total"].sum()
    eta0 = CoherentModel(cfg.with_(qe=0.0), cell).expected(1e5, 0.0)["mu_total"].sum()
    out.append(check("Edge cases N = 0, η = 0", e0 == 0 and eta0 == 0, f"Σμ(N=0)={e0}, Σμ(η=0)={eta0}", "0", "", "boundary"))
    step(0.8)
    samples = rng.poisson(5.0, size=200000)
    d = dispersion_test(samples, 5.0)
    lam = m.expected(1e5, cfg.noise_counts)["mu_total"]
    reps = np.stack([rng.poisson(lam) for _ in range(300)])
    fano = (reps.var(0, ddof=1) / reps.mean(0)).mean()
    out.append(check("Poisson statistics", abs(d["z_mean"]) < 4 and abs(d["z_var"]) < 4 and abs(fano - 1) < 0.02,
                     f"μ=5: mean z={d['z_mean']:.2f}, var z={d['z_var']:.2f}; pixel-mean Fano factor={fano:.4f}",
                     "|z|<4; Fano = 1 ± 0.02", "Counts have mean = variance as required for photon counting.", "statistics"))
    T = np.stack([m.sample(np.random.default_rng(k), N=1e5)["images"]["coherent"] for k in range(300)])
    zmap = (T.mean(0) - m.I_det) / (T.std(0, ddof=1) / np.sqrt(len(T)))
    out.append(check("Estimator unbiasedness", abs(zmap.mean()) < 0.1 and abs(zmap.std() - 1) < 0.1,
                     f"z-map mean={zmap.mean():.3f}, std={zmap.std():.3f}", "≈ N(0,1)", "", "statistics"))
    step(1.0)
    return out


def test_channel_1():
    assert_all(run_checks())
