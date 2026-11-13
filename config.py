"""Shared simulation configuration (one object for ALL channels).

Every channel receives the *same* SimConfig, so the synthetic cell, wavelength,
field of view, pixel size, detector efficiency, background flux, propagation
distance and photon-budget definition are identical by construction.

Units: lengths in micrometres (um), spatial frequency in cycles/um, wavenumber
in rad/um, photon numbers dimensionless.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace

import numpy as np

PHOTON_LEVELS = {
    "Düşük (10²)": 1e2,
    "Orta (10³)": 1e3,
    "Yüksek (10⁴)": 1e4,
    "Çok yüksek (10⁵)": 1e5,
}


@dataclass(frozen=True)
class SimConfig:
    # ---- optics --------------------------------------------------------
    wavelength_um: float = 0.810      # vacuum wavelength lambda0
    n_medium: float = 1.335           # surrounding medium
    NA: float = 0.40                  # imaging numerical aperture
    z_um: float = 0.0                 # propagation distance sample -> detector-conjugate plane
    A0: float = 1.0                   # incident field amplitude (cancels in photon normalisation)
    # ---- sampling --------------------------------------------------------
    grid: int = 128                   # optical grid points per axis
    fov_um: float = 64.0              # field of view
    bin_factor: int = 2               # detector pixel = bin x bin optical samples
    # ---- cell geometry ---------------------------------------------------
    cell_radius_um: float = 14.0
    cell_aspect: float = 0.80         # minor/major semi-axis
    nucleus_radius_um: float = 5.5
    thickness_um: float = 5.0         # maximum cell height
    membrane_um: float = 0.30
    # ---- material properties (ASSUMED, stained-cell-like) ------------------
    mu_cyto: float = 0.020            # absorption coefficient [1/um]
    mu_nuc: float = 0.100
    mu_mem: float = 0.600
    mu_gran: float = 0.300
    n_cyto: float = 1.355             # refractive indices (absolute)
    n_nuc: float = 1.375
    n_mem: float = 1.435
    n_gran: float = 1.385
    texture_amp: float = 0.35
    n_granules: int = 30
    cell_seed: int = 7
    # ---- detection / budget ------------------------------------------------
    photons: float = 1e5              # photon budget N
    qe: float = 0.70                  # detector quantum efficiency (all detectors)
    background: float = 1.0           # background counts / pixel / acquisition
    dark: float = 0.0                 # dark counts / pixel / acquisition
    budget_mode: str = "dose"         # 'dose' (N on sample) or 'total' (N in all arms)
    repeats: int = 3
    seed: int = 2026

    # ---- derived -------------------------------------------------------
    @property
    def dx(self) -> float:
        return self.fov_um / self.grid

    @property
    def dx_det(self) -> float:
        return self.dx * self.bin_factor

    @property
    def n_det(self) -> int:
        return self.grid // self.bin_factor

    @property
    def n_det_pixels(self) -> int:
        return self.n_det ** 2

    @property
    def k0(self) -> float:
        return 2 * np.pi / self.wavelength_um

    @property
    def wavelength_medium(self) -> float:
        return self.wavelength_um / self.n_medium

    @property
    def coherent_cutoff(self) -> float:
        return self.NA / self.wavelength_um

    @property
    def nyquist(self) -> float:
        return 1.0 / (2 * self.dx)

    @property
    def noise_counts(self) -> float:
        """Background + dark counts per detector pixel over the acquisition."""
        return self.background + self.dark

    def with_(self, **kw) -> "SimConfig":
        return replace(self, **kw)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(dx_um=self.dx, detector_pixel_um=self.dx_det, n_det=self.n_det,
                 k0_rad_per_um=self.k0, coherent_cutoff_cyc_per_um=self.coherent_cutoff)
        return d

    def sampling_warnings(self) -> list[str]:
        w = []
        if self.coherent_cutoff > self.nyquist:
            w.append(f"Pupil cut-off NA/λ = {self.coherent_cutoff:.2f} c/µm exceeds the grid Nyquist "
                     f"{self.nyquist:.2f} c/µm: the pupil does not filter; resolution is grid-limited.")
        if self.cell_radius_um > 0.45 * self.fov_um:
            w.append("Cell is close to the FOV boundary: periodic boundary effects possible.")
        if self.nucleus_radius_um > 0.7 * self.cell_radius_um * self.cell_aspect:
            w.append("Nucleus nearly as large as the cell's minor axis; geometry clipped.")
        return w


def photon_split(cfg: SimConfig, channel: str) -> dict:
    """Photon budget per arm under the selected budget mode.

    dose  : N photons (or N pairs) incident on the SAMPLE in every channel.
    total : N photons generated in ALL arms together.
    """
    N = cfg.photons
    if channel == "ch1":
        return dict(sample=N, reference=0.0, pairs=0.0, total=N)
    if channel == "ch2":
        s = N if cfg.budget_mode == "dose" else N / 2
        return dict(sample=s, reference=s, pairs=0.0, total=2 * s)
    s = N if cfg.budget_mode == "dose" else N / 2      # pairs
    return dict(sample=s, reference=s, pairs=s, total=2 * s)
