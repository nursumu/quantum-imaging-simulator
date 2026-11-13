"""Image-quality metrics (identical definitions for all channels).

Reference image  T_ref(X): pixel-average of exp(-tau)  (object intensity transmittance).
System image     T_sys(X): expectation of a channel's estimator with infinite photons.

SNR_noise  = mean_cell(T_sys) / RMS_cell(T_hat - T_sys)     noise only
SNR_object = mean_cell(T_ref) / RMS_cell(T_hat - T_ref)     noise + systematic error
CNR        = |<T_hat>_nucleus - <T_hat>_cytoplasm| / sqrt((var_n + var_c)/2)
SSIM       = Wang et al. (2004) structural similarity vs T_ref, Gaussian window
             sigma = 1.5 px, K1 = 0.01, K2 = 0.03, data range 1. This is the same
             definition used by Fiji/ImageJ SSIM plugins; it is computed here with
             scikit-image so that the app runs without a Fiji installation.
Resolution = 10-90 % width of the edge-spread function (ESF) of an absorbing
             strip imaged through the same pipeline and photon budget; ESF =
             row-average, fitted with an error function: width = 2.563 sigma.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc
from skimage.metrics import structural_similarity

from .propagation import bin_mean


def detector_masks(cell: dict, b: int) -> dict:
    return {k: bin_mean(v.astype(float), b) >= 0.5 for k, v in cell["masks"].items()}


def reference_transmittance(tau: np.ndarray, b: int) -> np.ndarray:
    return bin_mean(np.exp(-tau), b)


def _clean(img):
    img = np.asarray(img, dtype=float)
    return np.where(np.isfinite(img), img, 0.0)


def snr(est, target, roi):
    est, target = _clean(est), _clean(target)
    rms = np.sqrt(np.mean((est[roi] - target[roi]) ** 2))
    return float(np.mean(target[roi]) / rms) if rms > 0 else np.inf


def cnr(est, m1, m2):
    est = _clean(est)
    a, b = est[m1], est[m2]
    if a.size == 0 or b.size == 0:
        return np.nan
    den = np.sqrt((a.var() + b.var()) / 2)
    return float(abs(a.mean() - b.mean()) / den) if den > 0 else np.inf


def ssim(est, ref):
    return float(structural_similarity(_clean(ref), _clean(est), data_range=1.0, gaussian_weights=True,
                                       sigma=1.5, use_sample_covariance=False))


def image_metrics(est, ref, sys_img, masks) -> dict:
    roi = masks["cell"]
    defined = bool(np.isfinite(np.asarray(est, float)).any())
    return dict(SNR_noise=snr(est, sys_img, roi), SNR_object=snr(est, ref, roi),
                CNR=cnr(est, masks["nucleus"], masks["cytoplasm"]), SSIM=ssim(est, ref),
                RMSE_object=float(np.sqrt(np.mean((_clean(est)[roi] - ref[roi]) ** 2))),
                estimator_defined=defined)


def _esf(x, lo, hi, x0, s):
    return lo + (hi - lo) * 0.5 * erfc((x - x0) / (np.sqrt(2) * s))


def edge_resolution(img, dx_det: float, edge_x_um: float = 0.0, half_window_um: float = 8.0,
                    expected_step: float = 0.9) -> dict:
    """10-90 % ESF width; NaN when not resolvable (fit failed, >50 % uncertainty,
    step height off by >1/3, or width below half a pixel)."""
    img = _clean(img)
    n = img.shape[1]
    x = (np.arange(n) - n / 2 + 0.5) * dx_det
    prof = img.mean(axis=0)
    sel = np.abs(x - edge_x_um) <= half_window_um
    xs, ys = x[sel], prof[sel]
    out = dict(width_10_90_um=np.nan, width_err_um=np.nan, x=x, profile=prof, fit=None)
    try:
        p0 = [ys[xs > edge_x_um].mean(), ys[xs < edge_x_um].mean(), edge_x_um, 0.7]
        popt, pcov = curve_fit(_esf, xs, ys, p0=p0, bounds=([-2, -2, edge_x_um - 3, 0.05],
                                                            [3, 3, edge_x_um + 3, 6]), maxfev=20000)
        s, ds = abs(popt[3]), float(np.sqrt(abs(pcov[3, 3])))
        w, dw = 2.5631 * s, 2.5631 * ds
        step = popt[1] - popt[0]
        ok = (np.isfinite(dw) and dw < 0.5 * w and step > 0 and abs(step - expected_step) < expected_step / 3
              and w >= 0.5 * dx_det)
        out["fit"] = _esf(x, *popt)
        if ok:
            out.update(width_10_90_um=float(w), width_err_um=float(dw))
    except Exception:
        pass
    return out


def summarize(runs: list[dict]) -> dict:
    keys = [k for k, v in runs[0].items() if np.isscalar(v) and not isinstance(v, (str, bool))]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in runs], dtype=float)
        f = vals[np.isfinite(vals)]
        out[k] = float(f.mean()) if f.size else np.nan
        out[k + "_std"] = float(f.std(ddof=1)) if f.size > 1 else 0.0
    return out
