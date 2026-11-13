"""CHANNEL 3 - Entangled-photon coincidence imaging (Dolanık Fotonlarla Kuantum Görüntüleme).

1. Biphoton state (SPDC, thin crystal, wide Gaussian pump; double-Gaussian
   approximation, Law & Eberly PRL 92, 127903 (2004)):
       |Psi> = ∫∫ dx_s dx_i Psi(x_s, x_i) |x_s>|x_i>   (per transverse axis; 2-D = product of x and y)
       Psi(x_s, x_i) = C exp[ -(x_s + x_i)^2 / (4 sigma_S^2) - (x_s - x_i)^2 / (4 sigma_D^2) ]
   sigma_S ~ pump waist, sigma_D ~ sqrt(L_crystal lambda) (phase matching).
   |Psi|^2 gives Var(x_s + x_i) = sigma_S^2, Var(x_s - x_i) = sigma_D^2. In momentum
   space Var(k_s + k_i) = 1/sigma_S^2. Schmidt number per axis K = (sigma_S/sigma_D + sigma_D/sigma_S)/2.
   Entanglement witness (Reid EPR): Delta^2(x_s - x_i) Delta^2(k_s + k_i) = sigma_D^2/sigma_S^2 < 1/4.
   Signal photon: goes through the sample. Idler photon: reference arm, never touches it.

2. Imaging geometry (near-field, heralded camera): crystal plane relayed to the sample
   (signal) and to the idler camera. Given an idler detection at x_i the signal is in the
   conditional state psi(x_s|x_i) ~ Psi(x_s, x_i) (packet of rms sigma_D at x_i; the pump
   is treated as uniform over the FOV, error < 2 % for sigma_S >= 400 um).
3. Sample interaction at the AMPLITUDE level, then imaging:
       v(x|x_i) = F^-1{ H P F[ t psi(.|x_i) ] },   P(X_s, x_i) = sum_{x in X_s} |v|^2  (<= 1)
   absorption removes probability, phase gradients redirect it (possibly out of the pupil).
4.-7. Detection efficiencies eta_s, eta_i; losses; dark & background counts on both cameras.
8.-9. True and accidental coincidences in M = N_pairs / mu windows (event Monte Carlo).
10. Reconstruction T_hat = (C - A_delayed) / (eta_s eta_i N_pairs K_flat).

Separable control: a mixture of product states with the SAME near-field position
correlation (each photon a packet of rms sigma_D/sqrt(2) around a common random X).
It has no momentum anti-correlation (EPR product ~ 1 >= 1/4).
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import uniform_filter

from calculations.correlations import count_coincidences, frame_photon_number_correlations
from calculations.propagation import system_transfer
from calculations.reconstruction import coincidence_transmittance
from core.config import SimConfig, photon_split


# ----------------------------------------------------------- analytics -----
def epr_analytic(sigma_D, sigma_S):
    a, b = 1 / (4 * sigma_S ** 2), 1 / (4 * sigma_D ** 2)
    return dict(var_x_minus=1 / (4 * b), var_k_plus=4 * a, product=a / b,
                schmidt_per_axis=0.5 * (np.sqrt(a / b) + np.sqrt(b / a)))


def biphoton_density(xs, xi, sigma_D, sigma_S, state="entangled"):
    if state == "entangled":
        return np.exp(-(xs + xi) ** 2 / (2 * sigma_S ** 2) - (xs - xi) ** 2 / (2 * sigma_D ** 2))
    return np.exp(-(xs - xi) ** 2 / (2 * sigma_D ** 2))


def biphoton_density_k(ks, ki, sigma_D, sigma_S, state="entangled"):
    if state == "entangled":
        return np.exp(-(ks + ki) ** 2 * sigma_S ** 2 / 2 - (ks - ki) ** 2 * sigma_D ** 2 / 2)
    w = sigma_D / np.sqrt(2)
    return np.exp(-(ks ** 2 + ki ** 2) * 2 * w ** 2)


def sample_pairs_1d(rng, n, state, sigma_D, sigma_S, space="x"):
    if state == "entangled":
        if space == "x":
            S, D = rng.normal(0, sigma_S, n), rng.normal(0, sigma_D, n)
        else:
            S, D = rng.normal(0, 1 / sigma_S, n), rng.normal(0, 1 / sigma_D, n)
        return (S + D) / 2, (S - D) / 2
    w = sigma_D / np.sqrt(2)
    if space == "x":
        X = rng.uniform(-sigma_S, sigma_S, n)
        return X + rng.normal(0, w, n), X + rng.normal(0, w, n)
    return rng.normal(0, 1 / (2 * w), n), rng.normal(0, 1 / (2 * w), n)


def measured_epr_product(rng, n, state, sigma_D, sigma_S, dx_near=1.0, dk_far=0.005, n_boot=200):
    xs, xi = sample_pairs_1d(rng, n, state, sigma_D, sigma_S, "x")
    ks, ki = sample_pairs_1d(rng, n, state, sigma_D, sigma_S, "k")
    q = lambda v, d: (np.floor(v / d) + 0.5) * d
    dm, kp = q(xs, dx_near) - q(xi, dx_near), q(ks, dk_far) + q(ki, dk_far)
    boots = [dm[rng.integers(n, size=n)].var() * kp[rng.integers(n, size=n)].var() for _ in range(n_boot)]
    return float(dm.var() * kp.var()), np.percentile(boots, [2.5, 97.5])


# --------------------------------------------------- conditional kernel ---
def patch_size(cfg: SimConfig) -> int:
    P = 4 * int(round(6.0 / cfg.dx))
    return int(np.clip(P, 16, min(64, cfg.grid)))


def _packet(P, cy, cx, dx, width, oy=0.0, ox=0.0):
    c = np.arange(P) * dx
    g = np.exp(-(((c - cy * dx - oy)[:, None]) ** 2 + ((c - cx * dx - ox)[None, :]) ** 2) / (4 * width ** 2))
    return g / np.sqrt(np.sum(g ** 2))


def conditional_kernel(cfg: SimConfig, t, sigma_D, state="entangled", chunk=1024, progress=None):
    """P(signal detector cell | idler optical position), window origins in detector pixels."""
    n, b = cfg.grid, cfg.bin_factor
    P = patch_size(cfg)
    h = P // 2
    nw = P // b
    tpad = np.pad(t, h, mode="wrap")
    win = sliding_window_view(tpad, (P, P))
    Hs = system_transfer(cfg, n=P, dx=cfg.dx)
    if state == "entangled":
        quad, width = [(0.0, 0.0, 1.0)], sigma_D
    else:
        width = sigma_D / np.sqrt(2)
        nodes, wts = np.array([-np.sqrt(3), 0, np.sqrt(3)]) * width, np.array([1 / 6, 2 / 3, 1 / 6])
        quad = [(nodes[i], nodes[j], wts[i] * wts[j]) for i in range(3) for j in range(3)]
    iy, ix = np.divmod(np.arange(n * n), n)
    sy, sx = iy - h + (iy % 2), ix - h + (ix % 2)
    Pc = np.zeros((n * n, nw * nw), np.float32)
    done, total = 0, n * n
    for py in (0, 1):
        for px in (0, 1):
            sel = np.where((iy % 2 == py) & (ix % 2 == px))[0]
            packets = [(_packet(P, h - py, h - px, cfg.dx, width, oy, ox), wq) for oy, ox, wq in quad]
            for k in range(0, len(sel), chunk):
                s = sel[k:k + chunk]
                tp = win[sy[s] + h, sx[s] + h]
                acc = np.zeros((len(s), nw * nw))
                for g, wq in packets:
                    I = np.abs(np.fft.ifft2(Hs * np.fft.fft2(tp * g))) ** 2
                    acc += wq * I.reshape(len(s), nw, b, nw, b).sum(axis=(2, 4)).reshape(len(s), -1)
                Pc[s] = acc
                done += len(s)
                if progress:
                    progress(done / total)
    return Pc, sy // b, sx // b


def coincidence_expectation(cfg, Pc, y0, x0, r):
    n, b, nd = cfg.grid, cfg.bin_factor, cfg.n_det
    nw = int(round(np.sqrt(Pc.shape[1])))
    iy, ix = np.divmod(np.arange(n * n), n)
    Yi, Xi = iy // b, ix // b
    cy, cx = np.divmod(np.arange(nw * nw), nw)
    Ys, Xs = y0[:, None] + cy[None, :], x0[:, None] + cx[None, :]
    on = (Ys >= 0) & (Ys < nd) & (Xs >= 0) & (Xs < nd)
    inW = on & (np.abs(Ys - Yi[:, None]) <= r) & (np.abs(Xs - Xi[:, None]) <= r)
    flat = np.where(on, Ys * nd + Xs, 0)
    Ks = np.bincount(flat[on], weights=Pc[on], minlength=nd * nd).reshape(nd, nd) / (n * n)
    K = np.bincount(flat[inW], weights=Pc[inW], minlength=nd * nd).reshape(nd, nd) / (n * n)
    return K, Ks


class EntangledPhotonModel:
    name = "ch3"

    def __init__(self, cfg: SimConfig, obj: dict, sigma_D=0.8, sigma_S=400.0, mu=0.1, window_r=1,
                 state="entangled", flat=None, progress=None):
        self.cfg, self.sigma_D, self.sigma_S = cfg, max(float(sigma_D), 0.75 * cfg.dx), float(sigma_S)
        self.mu, self.r, self.state = float(mu), int(window_r), state
        self.Pc, self.y0, self.x0 = conditional_kernel(cfg, obj["t"], self.sigma_D, state, progress=progress)
        self.K, self.Ks = coincidence_expectation(cfg, self.Pc, self.y0, self.x0, self.r)
        if flat is None:
            Pf, yf, xf = conditional_kernel(cfg, np.ones_like(obj["t"]), self.sigma_D, state)
            flat = coincidence_expectation(cfg, Pf, yf, xf, self.r)
        self.K_flat, self.Ks_flat = flat
        self.sys = {"coincidence": self.K / self.K_flat, "singles": self.Ks / self.Ks_flat}
        self.nw = int(round(np.sqrt(self.Pc.shape[1])))
        cdf = np.cumsum(self.Pc.astype(np.float64), axis=1)
        cdf = np.concatenate([cdf, np.ones((cdf.shape[0], 1))], axis=1)
        cdf[:, -1] = 1.0
        self.ncell = self.nw * self.nw + 1
        self.flat_cdf = (cdf + np.arange(cdf.shape[0])[:, None]).ravel()
        self.signal_transmission = float(self.Pc.sum() / self.Pc.shape[0])

    def events(self, N_pairs, noise, rng, mu=None, eta_s=None, eta_i=None):
        c = self.cfg
        mu = self.mu if mu is None else mu
        eta_s = c.qe if eta_s is None else eta_s
        eta_i = c.qe if eta_i is None else eta_i
        nd, npx = c.n_det, c.n_det_pixels
        M = max(2, int(round(N_pairs / mu)))
        Ng = rng.poisson(N_pairs)
        iopt = rng.integers(c.grid ** 2, size=Ng)
        cell = np.searchsorted(self.flat_cdf, iopt + rng.random(Ng), side="right") - iopt * self.ncell
        cell = np.minimum(cell, self.ncell - 1)
        lost = cell == self.ncell - 1
        cy, cx = np.divmod(np.where(lost, 0, cell), self.nw)
        Ys, Xs = self.y0[iopt] + cy, self.x0[iopt] + cx
        on = (~lost) & (Ys >= 0) & (Ys < nd) & (Xs >= 0) & (Xs < nd)
        det_s = on & (rng.random(Ng) < eta_s)
        det_i = rng.random(Ng) < eta_i
        frame = rng.integers(M, size=Ng)
        iy, ix = np.divmod(iopt, c.grid)
        pix_i = (iy // c.bin_factor) * nd + ix // c.bin_factor
        pix_s = np.where(on, Ys * nd + Xs, 0)
        nbs, nbi = rng.poisson(noise * npx), rng.poisson(noise * npx)
        ids = np.arange(Ng)
        sig = dict(pix=np.concatenate([pix_s[det_s], rng.integers(npx, size=nbs)]),
                   frame=np.concatenate([frame[det_s], rng.integers(M, size=nbs)]),
                   pair=np.concatenate([ids[det_s], -np.ones(nbs, int)]))
        idl = dict(pix=np.concatenate([pix_i[det_i], rng.integers(npx, size=nbi)]),
                   frame=np.concatenate([frame[det_i], rng.integers(M, size=nbi)]),
                   pair=np.concatenate([ids[det_i], -np.ones(nbi, int)]))
        budget = dict(pairs_generated=int(Ng), signal_lost_sample_or_pupil=int(lost.sum()),
                      signal_off_sensor=int((~lost & ~on).sum()), signal_reaching_detector=int(on.sum()),
                      signal_detected=int(det_s.sum()), idler_detected=int(det_i.sum()),
                      signal_noise_counts=int(nbs), idler_noise_counts=int(nbi),
                      both_detected=int((det_s & det_i).sum()), windows=M)
        return sig, idl, budget

    def sample(self, rng, N=None, noise=None, bg_factor=1.0, mode=None, mu=None, eta_s=None, eta_i=None,
               keep=False) -> dict:
        cfg = self.cfg if mode is None else self.cfg.with_(budget_mode=mode)
        cfg = cfg if N is None else cfg.with_(photons=N)
        noise = cfg.noise_counts if noise is None else noise
        eta_s = cfg.qe if eta_s is None else eta_s
        eta_i = cfg.qe if eta_i is None else eta_i
        Np = photon_split(cfg, "ch3")["pairs"]
        nd, mn = cfg.n_det, cfg.n_det_pixels
        sig, idl, bud = self.events(Np, noise, rng, mu, eta_s, eta_i)
        M = bud["windows"]
        cnt, tru = count_coincidences(sig, idl, nd, M, self.r, 0)
        cntd, _ = count_coincidences(sig, idl, nd, M, self.r, 1)
        img = lambda w: np.bincount(sig["pix"], weights=w, minlength=mn).reshape(nd, nd)
        C, Ct, A = img(cnt), img(tru), img(cntd)
        Ss = np.bincount(sig["pix"], minlength=mn).reshape(nd, nd).astype(float)
        Si = np.bincount(idl["pix"], minlength=mn).reshape(nd, nd).astype(float)
        T_c, norm = coincidence_transmittance(C, A, eta_s, eta_i, Np, self.K_flat)
        T_s = (Ss - bg_factor * noise) / (eta_s * Np * self.Ks_flat)
        k = 2 * self.r + 1
        SiW = uniform_filter(Si, size=k, mode="constant") * k * k
        with np.errstate(divide="ignore", invalid="ignore"):
            g2 = M * C / (Ss * SiW)
        pn = frame_photon_number_correlations(sig["frame"], idl["frame"], M)
        n_true, n_tot, n_dly = tru.sum(), cnt.sum(), cntd.sum()
        sysT = self.sys["coincidence"]
        unc = np.sqrt(norm * np.clip(sysT, 0, None) + 2 * max(n_dly, 0) / mn) / norm
        cellm = sysT < 0.999
        extras = dict(bud, N_pairs=Np, N_incident=Np, N_reference=Np, N_generated_total=2 * Np, frames=M,
                      detected_total=float(len(sig["pix"]) + len(idl["pix"])),
                      coincidences_total=float(n_tot), coincidences_true=float(n_true),
                      coincidences_accidental=float(n_tot - n_true), accidentals_delayed=float(n_dly),
                      CAR=float(n_true / max(n_tot - n_true, 1e-9)),
                      CAR_est=float((n_tot - n_dly) / max(n_dly, 1e-9)),
                      expected_true=float(eta_s * eta_i * Np * self.K.sum()),
                      coinc_rel_uncertainty=float(np.median((unc / np.maximum(sysT, 1e-6))[cellm])) if cellm.any() else np.nan,
                      **pn)
        out = dict(images={"coincidence": T_c, "singles": T_s}, extras=extras)
        if keep:
            out.update(maps=dict(C=C, C_true=Ct, A=A, S_s=Ss, S_i=Si, g2=g2, unc=unc,
                                 G2=C / M), events=(sig, idl))
        return out


def joint_row_matrix(sig, idl, nd, row):
    """Coincidences (no spatial window) between signal x and idler x for one detector row."""
    ss = (sig["pix"] // nd) == row
    ii = np.abs(idl["pix"] // nd - row) <= 1
    fs, xs = sig["frame"][ss], sig["pix"][ss] % nd
    o = np.argsort(idl["frame"][ii], kind="stable")
    fi, xi = idl["frame"][ii][o], (idl["pix"][ii] % nd)[o]
    lo, hi = np.searchsorted(fi, fs, "left"), np.searchsorted(fi, fs, "right")
    J = np.zeros((nd, nd))
    kmax = int((hi - lo).max()) if len(lo) else 0
    for k in range(kmax):
        j = lo + k
        v = j < hi
        np.add.at(J, (xs[v], xi[j[v]]), 1)
    return J
