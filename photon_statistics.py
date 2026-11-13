"""Photon counting statistics.

Photon detection by an ideal counting detector is a Poisson process: a pixel
whose expected count is mu registers N ~ Poisson(mu), with mean = variance = mu.

Coherent channel (specification, eq. F):
    mu_sig(i,j) = N_in * eta * |U_z(i,j)|^2 / sum_ij |U_in(i,j)|^2
summed over the optical samples belonging to a detector pixel. Because the
denominator is the INCIDENT power (not the transmitted power), absorption and
pupil losses reduce the expected counts: total losses are preserved.
    mu_total = mu_sig + background + dark
"""
from __future__ import annotations

import numpy as np

from .propagation import bin_sum


def coherent_expected_counts(Uz: np.ndarray, U_in: np.ndarray, N_in: float, eta: float,
                             bin_factor: int, noise_per_pixel: float) -> dict:
    norm = np.sum(np.abs(U_in) ** 2)
    mu_opt = N_in * eta * np.abs(Uz) ** 2 / norm
    mu_sig = bin_sum(mu_opt, bin_factor)
    return dict(mu_sig=mu_sig, mu_total=mu_sig + noise_per_pixel)


def poisson_counts(mu: np.ndarray, rng) -> np.ndarray:
    return rng.poisson(np.clip(mu, 0, None))


def dispersion_test(samples: np.ndarray, mu: float) -> dict:
    """Mean, variance and Fano factor of repeated counts; Poisson => Fano = 1."""
    m, v = samples.mean(), samples.var(ddof=1)
    n = samples.size
    se_mean = np.sqrt(mu / n)
    se_var = mu * np.sqrt(2 / (n - 1) + 1 / (mu * n)) if mu > 0 else 0.0
    return dict(mean=float(m), var=float(v), fano=float(v / m) if m > 0 else np.nan,
                z_mean=float((m - mu) / se_mean) if se_mean > 0 else 0.0,
                z_var=float((v - mu) / se_var) if se_var > 0 else 0.0)
