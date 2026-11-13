"""Shared synthetic biological cell (used identically by all three channels).

Geometry: a half-ellipsoidal dome on a flat substrate (semi-axes R and aspect*R,
height h) bounded by a membrane shell of thickness d_m; an ellipsoidal nucleus;
cytoplasm with smooth random heterogeneity; small granules (vesicles).

For each transverse point the geometric path length L_j(x, y) [um] through each
compartment j is computed (projection / thin-object approximation).

  absorption coefficient  mu_a,j        [1/um]   material property (assumed values)
  optical depth           tau(x,y) = sum_j mu_a,j L_j        dimensionless
  intensity transmittance T = exp(-tau)                      (Beer-Lambert)
  refractive index        n_j (absolute);  Delta n_j = n_j - n_medium
  phase delay             phi(x,y) = k0 sum_j Delta n_j L_j  [rad], k0 = 2 pi/lambda0
  complex transmission    t(x,y) = exp(-tau/2) exp(i phi)

alpha(x, y) in t = exp(-alpha/2) exp(i phi) is exactly the optical depth tau; the
factor 1/2 converts intensity attenuation into field-amplitude attenuation.

Limitations: thin-object approximation (no diffraction inside the cell) - valid
when h is small compared with the depth of field ~ lambda/NA^2; values of mu_a
and n are illustrative literature-range assumptions, not measurements.
"""
from __future__ import annotations

import numpy as np

from core.config import SimConfig


def grid_coords(cfg: SimConfig):
    c = (np.arange(cfg.grid) - cfg.grid / 2 + 0.5) * cfg.dx
    return np.meshgrid(c, c, indexing="xy")


def smooth_random_field(n: int, dx: float, corr_um: float, rng) -> np.ndarray:
    f = np.fft.fftfreq(n, d=dx)
    FX, FY = np.meshgrid(f, f, indexing="xy")
    filt = np.exp(-2 * (np.pi * corr_um) ** 2 * (FX ** 2 + FY ** 2))
    g = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * filt))
    s = g.std()
    return (g - g.mean()) / s if s > 0 else g


def _dome(rho2, height):
    return np.where(rho2 < 1.0, height * np.sqrt(np.clip(1.0 - rho2, 0, None)), 0.0)


def build_cell(cfg: SimConfig) -> dict:
    rng = np.random.default_rng(cfg.cell_seed)
    X, Y = grid_coords(cfg)
    th = np.deg2rad(20.0)
    Xr = X * np.cos(th) + Y * np.sin(th)
    Yr = -X * np.sin(th) + Y * np.cos(th)
    a = cfg.cell_radius_um
    b = cfg.cell_radius_um * cfg.cell_aspect
    h, d = cfg.thickness_um, cfg.membrane_um

    rho2_out = (Xr / a) ** 2 + (Yr / b) ** 2
    rho2_in = (Xr / max(a - d, 1e-3)) ** 2 + (Yr / max(b - d, 1e-3)) ** 2
    L_out = _dome(rho2_out, h)
    L_in = _dome(rho2_in, max(h - 2 * d, 0.0))
    L_mem = np.clip(L_out - L_in, 0, None)

    an = cfg.nucleus_radius_um
    bn = cfg.nucleus_radius_um * 0.82
    cx, cy = 0.18 * a, -0.12 * b
    Xn, Yn = Xr - cx, Yr - cy
    rho2_n = (Xn / an) ** 2 + (Yn / bn) ** 2
    L_nuc = np.minimum(_dome(rho2_n, 0.70 * h), L_in)

    L_gran = np.zeros_like(L_out)
    rg = 0.6
    centres, tries = [], 0
    while len(centres) < cfg.n_granules and tries < 20000:
        tries += 1
        gx, gy = rng.uniform(-a, a), rng.uniform(-b, b)
        if (gx / max(a - 2, 1)) ** 2 + (gy / max(b - 2, 1)) ** 2 > 1:
            continue
        if ((gx - cx) / (an + 1.0)) ** 2 + ((gy - cy) / (bn + 1.0)) ** 2 < 1:
            continue
        if any((gx - u) ** 2 + (gy - v) ** 2 < (2.5 * rg) ** 2 for u, v in centres):
            continue
        centres.append((gx, gy))
        r2 = (Xr - gx) ** 2 + (Yr - gy) ** 2
        L_gran += np.where(r2 < rg ** 2, 2 * np.sqrt(np.clip(rg ** 2 - r2, 0, None)), 0.0)
    L_gran = np.minimum(L_gran, np.clip(L_in - L_nuc, 0, None))
    L_cyto = np.clip(L_in - L_nuc - L_gran, 0, None)

    tex = smooth_random_field(cfg.grid, cfg.dx, 1.5, rng)
    mod = np.clip(1.0 + cfg.texture_amp * tex, 0.0, None)

    nm = cfg.n_medium
    mu = dict(cyto=cfg.mu_cyto * mod, nuc=cfg.mu_nuc, mem=cfg.mu_mem, gran=cfg.mu_gran)
    dn = dict(cyto=(cfg.n_cyto - nm) * mod, nuc=cfg.n_nuc - nm, mem=cfg.n_mem - nm, gran=cfg.n_gran - nm)
    L = dict(cyto=L_cyto, nuc=L_nuc, mem=L_mem, gran=L_gran)

    tau = sum(mu[k] * L[k] for k in L)
    opd = sum(dn[k] * L[k] for k in L)            # optical path difference [um]
    phi = cfg.k0 * opd
    t = np.exp(-tau / 2) * np.exp(1j * phi)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu_mean = np.where(L_out > 0, tau / L_out, 0.0)          # path-averaged mu_a
        n_mean = np.where(L_out > 0, nm + opd / L_out, nm)        # path-averaged index

    masks = dict(
        cell=rho2_out < 1.0,
        nucleus=rho2_n < 1.0,
        membrane=(rho2_out < 1.0) & (rho2_in >= 0.85 ** 2),
        cytoplasm=(rho2_in < 0.85 ** 2) & (((Xn / (an + 1.0)) ** 2 + (Yn / (bn + 1.0)) ** 2) > 1.0),
        background=rho2_out > 1.15 ** 2,
    )
    labels = np.zeros(X.shape, int)                 # 0 medium, 1 membrane, 2 cytoplasm, 3 nucleus, 4 granule
    labels[masks["cell"]] = 2
    labels[(L_mem > 0.5 * L_out) & masks["cell"]] = 1
    labels[rho2_n < 1.0] = 3
    labels[L_gran > 0] = 4
    return dict(tau=tau, phi=phi, t=t, opd=opd, mu_mean=mu_mean, n_mean=n_mean, labels=labels,
                L_total=L_out, L_mem=L_mem, L_nuc=L_nuc, L_cyto=L_cyto, L_gran=L_gran,
                masks=masks, n_granules=len(centres), X=X, Y=Y)


def uniform_object(cfg: SimConfig, T: float = 1.0, phase: float = 0.0) -> dict:
    tau = np.full((cfg.grid, cfg.grid), -np.log(T))
    phi = np.full_like(tau, phase)
    return dict(tau=tau, phi=phi, t=np.exp(-tau / 2) * np.exp(1j * phi))


def edge_target(cfg: SimConfig, T_dark: float = 0.1) -> dict:
    """Absorbing strip 0 <= x < FOV/3 (sharp amplitude edge at x = 0) for ESF resolution."""
    X, _ = grid_coords(cfg)
    inside = (X >= 0) & (X < cfg.fov_um / 3)
    tau = np.where(inside, -np.log(T_dark), 0.0)
    return dict(tau=tau, phi=np.zeros_like(tau), t=np.exp(-tau / 2) + 0j, T_dark=T_dark, edge_x_um=0.0)
