"""Image reconstruction: every channel estimates the intensity transmittance T(X).

Channel 1   T_hat = (N - b_assumed) / (eta N_in / N_pix)
Channel 2   conv  : (<n_s> - b/M) / (eta a_s)
            ratio : [(<n_s>-b/M)/(eta a_s)] / [(<n_r>-b/M)/(eta a_r)]
            cov   : (a_r/a_s) C_sr / D,  D = spatial mean of [Var_f(n_r) - <n_r>]
                    (shot-noise-corrected reference variance; needs no knowledge of b)
Channel 3   T_hat = (C - A_delayed) / (eta_s eta_i N_pairs K_flat)
            K_flat = expected true-coincidence probability per pair for t = 1.
"""
from __future__ import annotations

import numpy as np


def coherent_transmittance(counts, noise_assumed, eta, N_in, n_pix):
    return (counts - noise_assumed) / (eta * N_in / n_pix)


def correlation_transmittance(stats: dict, a_s: float, a_r: float, eta: float, beta: float,
                              bg_factor: float = 1.0) -> dict:
    conv_s = (stats["mean_s"] - bg_factor * beta) / (eta * a_s)
    conv_r = (stats["mean_r"] - bg_factor * beta) / (eta * a_r)
    ratio = conv_s / np.maximum(conv_r, 0.05)
    D = float(np.mean(stats["var_r"] - stats["mean_r"]))
    if D > 0:
        cov = (a_r / a_s) * stats["cov"] / D
        se = (a_r / a_s) * stats["se_cov"] / D
    else:
        cov = np.full_like(stats["cov"], np.nan)
        se = np.full_like(stats["cov"], np.nan)
    return dict(conv=conv_s, ratio=ratio, cov=cov, se_cov_T=se, D=D)


def coincidence_transmittance(C, A, eta_s, eta_i, N_pairs, K_flat):
    norm = eta_s * eta_i * N_pairs * K_flat
    return (C - A) / norm, norm
