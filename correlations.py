"""Correlation estimators.

Classical (Channel 2), from M frames of photon counts n_s(X,f), n_r(X,f):
    <n>_f        = (1/M) sum_f n(X,f)                          frame average
    Delta n      = n - <n>_f                                   fluctuation
    C_sr(X)      = 1/(M-1) sum_f Delta n_s Delta n_r           unbiased cross-covariance
    SE[C_sr](X)  = std_f(Delta n_s Delta n_r) / sqrt(M)        estimation uncertainty
    g2_sr(X)     = <n_s n_r>_f / (<n_s>_f <n_r>_f)             normalised correlation
Because the Poisson noise of two different cameras is independent,
E[C_sr] = eta^2 a_s a_r Cov(I_s, I_r) exactly (photon counting is a
doubly-stochastic Poisson process).

Quantum (Channel 3), from time-tagged events (pixel, window):
    coincidence counts C(X_s)   : number of (signal, idler) event pairs in the same window
                                  with the idler pixel inside W(X_s)
    G2(X_s, X_i)                : <n_s(X_s) n_i(X_i)>_window = C / M   (unnormalised)
    g2(X_s, X_i)                : G2 / (<n_s><n_i>)  = M C / (S_s S_i)
    accidentals A(X_s)          : same count with the idler stream delayed by one window
"""
from __future__ import annotations

import numpy as np


def frame_covariance(n_s: np.ndarray, n_r: np.ndarray) -> dict:
    M = n_s.shape[0]
    ms, mr = n_s.mean(0), n_r.mean(0)
    prod = (n_s - ms) * (n_r - mr)
    cov = prod.sum(0) / (M - 1)
    se = prod.std(0, ddof=1) / np.sqrt(M)
    with np.errstate(divide="ignore", invalid="ignore"):
        g2 = (n_s * n_r).mean(0) / (ms * mr)
    pooled = float((n_s * n_r).mean(0).sum() / (ms * mr).sum()) if (ms * mr).sum() > 0 else np.nan
    return dict(mean_s=ms, mean_r=mr, cov=cov, se_cov=se, g2=g2, g2_pooled=pooled,
                var_r=n_r.var(0, ddof=1))


def count_coincidences(sig: dict, idl: dict, n_det: int, M: int, r: int = 1, delay: int = 0):
    """Count coincidences for every signal event.

    sig / idl: dicts of arrays 'pix' (flat detector index), 'frame' (window index),
    'pair' (pair id or -1 for noise). Returns (counts per signal event, true-pair
    flags per signal event). 'True' means the matched idler is the partner photon.
    """
    keys = idl["frame"].astype(np.int64) * n_det * n_det + idl["pix"]
    order = np.argsort(keys, kind="stable")
    ks = keys[order]
    ipair = idl["pair"][order]
    sy, sx = np.divmod(sig["pix"], n_det)
    f = (sig["frame"].astype(np.int64) + delay) % M
    cnt = np.zeros(len(sy))
    tru = np.zeros(len(sy))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ty, tx = sy + dy, sx + dx
            ok = (ty >= 0) & (ty < n_det) & (tx >= 0) & (tx < n_det)
            key = f * n_det * n_det + np.where(ok, ty * n_det + tx, 0)
            lo = np.searchsorted(ks, key, "left")
            hi = np.searchsorted(ks, key, "right")
            c = (hi - lo) * ok
            cnt += c
            if delay == 0:
                has = ok & (c > 0) & (sig["pair"] >= 0)
                idx = np.where(has)[0]
                kmax = int(c[has].max()) if idx.size else 0
                for k in range(kmax):
                    j = lo[idx] + k
                    v = j < hi[idx]
                    tru[idx[v]] += ipair[j[v]] == sig["pair"][idx[v]]
    return cnt, tru


def frame_photon_number_correlations(sig_frames, idl_frames, M) -> dict:
    """Whole-sensor photon-number correlations per coincidence window (no M-length arrays)."""
    us, cs = np.unique(sig_frames, return_counts=True)
    ui, ci = np.unique(idl_frames, return_counts=True)
    _, a, b = np.intersect1d(us, ui, return_indices=True)
    ms, mi = cs.sum() / M, ci.sum() / M
    if ms == 0 or mi == 0:
        return dict(g2_si=np.nan, g2_ss=np.nan, g2_ii=np.nan)
    return dict(g2_si=float((cs[a] * ci[b]).sum() / M / (ms * mi)),
                g2_ss=float((cs * (cs - 1)).sum() / M / ms ** 2),
                g2_ii=float((ci * (ci - 1)).sum() / M / mi ** 2))
