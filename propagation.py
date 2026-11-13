"""Scalar wave propagation: Angular Spectrum Method (ASM) and imaging pupil.

Theory
------
A monochromatic scalar field in a homogeneous medium (index n) is a superposition
of plane waves exp[i(k_x x + k_y y + k_z z)] with k_x = 2 pi f_x, k_y = 2 pi f_y and

    k_z = sqrt(k^2 - (2 pi f_x)^2 - (2 pi f_y)^2),   k = 2 pi n / lambda0.

Free-space transfer function: H(f_x, f_y, z) = exp(i k_z z).
  * propagating (k_x^2 + k_y^2 <= k^2): k_z real -> |H| = 1 (phase only, energy conserved)
  * evanescent  (k_x^2 + k_y^2 >  k^2): k_z = i kappa -> H = exp(-kappa z), decays.
U_z = F^-1[ F(U_0) H ]  (exact within scalar optics, no paraxial approximation).

Numerical approximation
-----------------------
The discrete FFT imposes periodic boundaries; the band-limit of Matsushima &
Shimobaba (2009) suppresses aliasing of H for long z. Back-propagation of
evanescent waves (z < 0) is refused because it amplifies noise exponentially.

Imaging pupil: unit-magnification, aberration-free system with numerical aperture
NA has the coherent transfer function P = circ(f / (NA/lambda0)). Light outside
the pupil is lost (never renormalised).
"""
from __future__ import annotations

import numpy as np


def freq_grid(n: int, dx: float):
    f = np.fft.fftfreq(n, d=dx)
    return np.meshgrid(f, f, indexing="xy")


def kz_map(n: int, dx: float, wavelength_medium: float) -> np.ndarray:
    """Complex k_z [rad/um]; imaginary part > 0 marks evanescent components."""
    FX, FY = freq_grid(n, dx)
    k = 2 * np.pi / wavelength_medium
    return np.sqrt((k ** 2 - (2 * np.pi * FX) ** 2 - (2 * np.pi * FY) ** 2).astype(complex))


def asm_transfer(n: int, dx: float, wavelength_medium: float, z: float,
                 band_limit: bool = True, evanescent: str = "decay") -> np.ndarray:
    if z == 0:
        return np.ones((n, n), dtype=complex)
    kz = kz_map(n, dx, wavelength_medium)
    evan = kz.imag > 0
    if z < 0 and evanescent == "decay" and evan.any():
        raise ValueError("Backward propagation would amplify evanescent waves; use evanescent='drop'.")
    H = np.exp(1j * kz * z)
    if evanescent == "drop":
        H = np.where(evan, 0.0, H)
    if band_limit:
        FX, FY = freq_grid(n, dx)
        df = 1.0 / (n * dx)
        flim = 1.0 / (wavelength_medium * np.sqrt((2 * df * z) ** 2 + 1.0))
        H = H * ((np.abs(FX) <= flim) & (np.abs(FY) <= flim))
    return H


def coherent_pupil(n: int, dx: float, NA: float, wavelength_vac: float) -> np.ndarray:
    FX, FY = freq_grid(n, dx)
    return (np.sqrt(FX ** 2 + FY ** 2) <= NA / wavelength_vac).astype(float)


def propagate(U0: np.ndarray, dx: float, wavelength_medium: float, z: float, pad: int = 0,
              band_limit: bool = True, evanescent: str = "decay", extra=None) -> np.ndarray:
    n = U0.shape[-1]
    if pad:
        fill = np.mean(np.concatenate([U0[0], U0[-1], U0[:, 0], U0[:, -1]]))
        U = np.pad(U0, pad, mode="constant", constant_values=fill)
    else:
        U = U0
    H = asm_transfer(U.shape[-1], dx, wavelength_medium, z, band_limit, evanescent)
    if extra is not None:
        H = H * extra
    Uz = np.fft.ifft2(np.fft.fft2(U) * H)
    return Uz[pad:pad + n, pad:pad + n] if pad else Uz


def system_transfer(cfg, n: int | None = None, dx: float | None = None) -> np.ndarray:
    """ASM over z  x  coherent pupil (both Fourier multipliers)."""
    n = cfg.grid if n is None else n
    dx = cfg.dx if dx is None else dx
    return asm_transfer(n, dx, cfg.wavelength_medium, cfg.z_um) * \
        coherent_pupil(n, dx, cfg.NA, cfg.wavelength_um)


def bin_mean(img: np.ndarray, b: int) -> np.ndarray:
    *lead, ny, nx = img.shape
    return img.reshape(*lead, ny // b, b, nx // b, b).mean(axis=(-3, -1))


def bin_sum(img: np.ndarray, b: int) -> np.ndarray:
    *lead, ny, nx = img.shape
    return img.reshape(*lead, ny // b, b, nx // b, b).sum(axis=(-3, -1))
