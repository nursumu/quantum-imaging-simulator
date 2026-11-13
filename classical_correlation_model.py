"""CHANNEL 2 - Classically correlated imaging (Klasik Korelasyonlu Işıkla Görüntüleme).

Model choice: SPLIT PSEUDO-THERMAL (SPECKLE) LIGHT
---------------------------------------------------
1. Classical source : a laser scattered by a rotating ground-glass diffuser. Each
   frame f carries an independent, fully developed speckle field
       E_f = F^-1[ F(f) W_f ],  W_f circular complex Gaussian white noise,
       |F|^2 = S(f) ~ exp(-2 pi^2 l_c^2 f^2)  (field-correlation length l_c),
   normalised so <|E|^2> = 1. Such a field is circular complex-Gaussian
   (thermal): exponential intensity statistics and photon bunching g2(0) = 2.
2. Beams            : a beam splitter makes two copies (signal, reference).
3. Sample           : signal field E_f t.
4. Propagation      : both arms pass identical ASM x pupil optics.
5. Detection        : two pixel-registered photon-counting cameras.
6. Statistics       : n_a ~ Poisson(eta a_a I_a + (b+d)/M),  a_a = N_a/(M N_pix).
7. Correlation      : C_sr = <Delta n_s Delta n_r> over M frames.
8. Reconstruction   : calculations/reconstruction.py.

Why appropriate: it is the standard classical analogue of two-photon imaging
(Gatti et al. PRL 93, 093602 (2004); Valencia et al. PRL 94, 063601 (2005);
Ferri et al. PRL 94, 183602 (2005)). All intensity correlations follow from the
field coherence by the Gaussian moment theorem
    Cov[I_s(x), I_r(x')] = |<U_s(x) U_r*(x')>|^2,
which this module evaluates EXACTLY (mode sum) for the noise-free expectation.
These correlations are CLASSICAL (common random classical wave); there is no
entanglement and g2_sr <= 2 (Cauchy-Schwarz).

Correlation strength c: reference field E_r = sqrt(c) E + sqrt(1-c) E' with an
independent speckle E'; Cov(I_s, I_r) is proportional to c. c = 0 is the
uncorrelated-beams control.
"""
from __future__ import annotations

import numpy as np

from calculations.correlations import frame_covariance
from calculations.propagation import freq_grid, system_transfer, bin_mean
from calculations.reconstruction import correlation_transmittance
from core.config import SimConfig, photon_split


class SpeckleSource:
    def __init__(self, cfg: SimConfig, corr_um: float):
        n = cfg.grid
        FX, FY = freq_grid(n, cfg.dx)
        S = np.exp(-2 * (np.pi * corr_um) ** 2 * (FX ** 2 + FY ** 2))
        self.S = S / S.sum()
        self.F = np.sqrt(self.S) * n * n
        self.n = n

    def coeffs(self, rng, m):
        W = (rng.standard_normal((m, self.n, self.n)) + 1j * rng.standard_normal((m, self.n, self.n))) / np.sqrt(2)
        return self.F * W


class ClassicalCorrelationModel:
    name = "ch2"

    def __init__(self, cfg: SimConfig, obj: dict, frames: int = 100, corr_strength: float = 1.0,
                 speckle_um: float = 2.5, exact: bool = True):
        self.cfg, self.t = cfg, obj["t"]
        self.M, self.c, self.speckle_um = int(frames), float(corr_strength), speckle_um
        self.src = SpeckleSource(cfg, speckle_um)
        self.H = system_transfer(cfg)
        self.sys = {}
        if exact:
            self._exact()

    def _exact(self):
        """Ensemble moments via the Gaussian moment theorem (plane-wave mode sum)."""
        cfg, n, b, nd = self.cfg, self.cfg.grid, self.cfg.bin_factor, self.cfg.n_det
        S = self.src.S
        idx = np.argwhere(S > 1e-9 * S.max())
        m = np.arange(n)
        J_sr = np.zeros((nd * nd, b * b, b * b), complex)
        J_rr = np.zeros_like(J_sr)
        ms, mr = np.zeros((n, n)), np.zeros((n, n))
        for k in range(0, len(idx), 48):
            ch = idx[k:k + 48]
            ky, kx = ch[:, 0], ch[:, 1]
            w = S[ky, kx]
            e = np.exp(1j * 2 * np.pi * (ky[:, None, None] * m[None, :, None] + kx[:, None, None] * m[None, None, :]) / n)
            us = np.fft.ifft2(self.H[None] * np.fft.fft2(self.t[None] * e))
            ur = self.H[ky, kx][:, None, None] * e
            ms += np.einsum("c,cyx->yx", w, np.abs(us) ** 2)
            mr += np.einsum("c,cyx->yx", w, np.abs(ur) ** 2)
            blk = lambda u: u.reshape(len(ch), nd, b, nd, b).transpose(0, 1, 3, 2, 4).reshape(len(ch), nd * nd, b * b)
            bs, br = blk(us), blk(ur)
            J_sr += np.einsum("c,cpj,cpk->pjk", w, bs, br.conj())
            J_rr += np.einsum("c,cpj,cpk->pjk", w, br, br.conj())
        nb = (b * b) ** 2
        self.cov_exact = (np.abs(J_sr) ** 2).sum((1, 2)).reshape(nd, nd) / nb
        self.var_r_exact = (np.abs(J_rr) ** 2).sum((1, 2)).reshape(nd, nd) / nb
        self.mean_s_exact = bin_mean(ms, b)
        self.mean_r_exact = bin_mean(mr, b)
        self.sys = {"conv": self.mean_s_exact, "ratio": self.mean_s_exact / self.mean_r_exact,
                    "cov": self.c * self.cov_exact / self.var_r_exact.mean()}

    def variant(self, corr_strength=None, frames=None):
        """Shallow copy with another correlation strength / frame number (reuses exact moments)."""
        import copy
        m = copy.copy(self)
        if corr_strength is not None:
            m.c = float(corr_strength)
            if self.sys:
                m.sys = dict(self.sys, cov=m.c * self.cov_exact / self.var_r_exact.mean())
        if frames is not None:
            m.M = int(frames)
        return m

    def frames(self, M, rng, chunk=32, keep_fields=False):
        b = self.cfg.bin_factor
        Is, Ir, Es, Er = [], [], None, None
        for k in range(0, M, chunk):
            m = min(chunk, M - k)
            A = self.src.coeffs(rng, m)
            E = np.fft.ifft2(A)
            A_r = A if self.c >= 1 else np.sqrt(self.c) * A + np.sqrt(1 - self.c) * self.src.coeffs(rng, m)
            Us = np.fft.ifft2(self.H * np.fft.fft2(E * self.t))
            Ur = np.fft.ifft2(self.H * A_r)
            Is.append(bin_mean(np.abs(Us) ** 2, b).astype(np.float32))
            Ir.append(bin_mean(np.abs(Ur) ** 2, b).astype(np.float32))
            if keep_fields and Es is None:
                Es, Er = E[0], np.fft.ifft2(A_r[0])
        out = dict(I_s=np.concatenate(Is), I_r=np.concatenate(Ir))
        if keep_fields:
            out.update(E_source=Es, E_ref_source=Er)
        return out

    def sample(self, rng, N=None, noise=None, bg_factor=1.0, mode=None, M=None, keep=False) -> dict:
        cfg = self.cfg if mode is None else self.cfg.with_(budget_mode=mode)
        cfg = cfg if N is None else cfg.with_(photons=N)
        noise = cfg.noise_counts if noise is None else noise
        M = self.M if M is None else int(M)
        sp = photon_split(cfg, "ch2")
        a_s, a_r = sp["sample"] / (M * cfg.n_det_pixels), sp["reference"] / (M * cfg.n_det_pixels)
        fr = self.frames(M, rng, keep_fields=keep)
        beta = noise / M
        n_s = rng.poisson(cfg.qe * a_s * fr["I_s"].astype(float) + beta)
        n_r = rng.poisson(cfg.qe * a_r * fr["I_r"].astype(float) + beta)
        st = frame_covariance(n_s, n_r)
        rec = correlation_transmittance(st, a_s, a_r, cfg.qe, beta, bg_factor)
        defined = rec["D"] > 0
        sysc = self.sys.get("cov", np.ones_like(st["cov"]))
        extras = dict(frames=M, N_incident=sp["sample"], N_reference=sp["reference"],
                      N_generated_total=sp["total"],
                      expected_signal_counts=float(cfg.qe * a_s * M * self.mean_s_exact.sum()) if self.sys else np.nan,
                      detected_signal=float(n_s.sum()), detected_reference=float(n_r.sum()),
                      detected_total=float(n_s.sum() + n_r.sum()),
                      ref_variance_D=rec["D"], cov_estimator_defined=float(defined),
                      cov_rel_uncertainty=float(np.median(rec["se_cov_T"] / np.maximum(np.abs(sysc), 1e-6)))
                      if defined else np.nan,
                      g2_sr_pooled=st["g2_pooled"],
                      corr_coeff=float(np.corrcoef(n_s.reshape(M, -1).sum(1), n_r.reshape(M, -1).sum(1))[0, 1])
                      if M > 2 and n_s.sum() > 0 and n_r.sum() > 0 else np.nan)
        out = dict(images={"conv": rec["conv"], "ratio": rec["ratio"], "cov": rec["cov"]}, extras=extras)
        if keep:
            out.update(stats=st, se_cov_T=rec["se_cov_T"], frame0=dict(I_s=fr["I_s"][0], I_r=fr["I_r"][0],
                       n_s=n_s[0], n_r=n_r[0], E_source=fr["E_source"], E_ref_source=fr["E_ref_source"]))
        return out
