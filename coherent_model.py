"""CHANNEL 1 - Classical coherent intensity imaging (Klasik Koherent Işık Şiddeti Görüntüleme).

A. Illumination      U_in(x,y) = A0                       monochromatic, spatially coherent plane wave
B. Sample            t(x,y) = exp(-alpha/2) exp(i phi)    (models/cell_model.py)
C. Exit field        U_0 = U_in t
D. Propagation       U_z = F^-1[ F(U_0) H(fx,fy,z) ] x pupil P (calculations/propagation.py)
E. Intensity         I = |U_z|^2
F. Expected counts   mu_sig(i,j) = N_in eta |U_z(i,j)|^2 / sum|U_in|^2 ;  mu_total = mu_sig + b + d
G. Counts            N(i,j) ~ Poisson(mu_total(i,j))
Reconstruction       T_hat = (N - b - d) / (eta N_in / N_pix)
"""
from __future__ import annotations

import numpy as np

from calculations.photon_statistics import coherent_expected_counts, poisson_counts
from calculations.propagation import system_transfer, bin_mean
from calculations.reconstruction import coherent_transmittance
from core.config import SimConfig, photon_split


class CoherentModel:
    name = "ch1"

    def __init__(self, cfg: SimConfig, obj: dict):
        self.cfg = cfg
        n = cfg.grid
        self.U_in = np.full((n, n), cfg.A0, dtype=complex)
        self.t = obj["t"]
        self.U0 = self.U_in * self.t
        self.F_U0 = np.fft.fft2(self.U0)
        self.Uz = np.fft.ifft2(self.F_U0 * system_transfer(cfg))
        self.I_opt = np.abs(self.Uz) ** 2
        self.I_det = bin_mean(self.I_opt, cfg.bin_factor) / abs(cfg.A0) ** 2   # = 1 for empty field
        self.sys = {"coherent": self.I_det}
        self.trans_sample = float(np.mean(np.abs(self.t) ** 2))

    def expected(self, N_in: float, noise: float) -> dict:
        c = self.cfg
        return coherent_expected_counts(self.Uz, self.U_in, N_in, c.qe, c.bin_factor, noise)

    def sample(self, rng, N=None, noise=None, bg_factor=1.0, mode=None) -> dict:
        c = self.cfg
        if mode is not None:
            c = c.with_(budget_mode=mode)
        N_in = photon_split(c.with_(photons=c.photons if N is None else N), "ch1")["sample"]
        noise = c.noise_counts if noise is None else noise
        e = self.expected(N_in, noise)
        counts = poisson_counts(e["mu_total"], rng)
        T_hat = coherent_transmittance(counts, bg_factor * noise, c.qe, N_in, c.n_det_pixels)
        extras = dict(N_incident=N_in, N_generated_total=N_in,
                      N_after_sample=N_in * self.trans_sample,
                      N_after_pupil=N_in * float(self.I_det.mean()),
                      expected_signal_counts=float(e["mu_sig"].sum()),
                      expected_noise_counts=float(noise * c.n_det_pixels),
                      detected_total=float(counts.sum()),
                      frames=1)
        return dict(images={"coherent": T_hat}, counts=counts, mu_sig=e["mu_sig"],
                    mu_total=e["mu_total"], extras=extras)
