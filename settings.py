"""Global simulation settings panel (sidebar) -> one SimConfig shared by every page."""
from __future__ import annotations

from dataclasses import asdict

import streamlit as st

from core.config import SimConfig, PHOTON_LEVELS

D = SimConfig()


def _num(label, key, lo, hi, default, step, fmt=None, help=None):
    return st.slider(label, lo, hi, st.session_state.get(key, default), step, format=fmt, key=key, help=help)


def global_settings() -> SimConfig:
    sb = st.sidebar
    sb.markdown("### Genel simülasyon ayarları")
    sb.caption("Tüm kanallar bu ortak parametreleri kullanır (aynı hücre, aynı optik, aynı foton bütçesi tanımı).")
    with sb.expander("Optik ve örnekleme", expanded=False):
        lam = _num("Wavelength λ₀ (µm)", "lam", 0.40, 1.10, D.wavelength_um, 0.01)
        NA = _num("Numerical aperture NA", "NA", 0.10, 1.00, D.NA, 0.01)
        z = _num("Propagation distance z (µm)", "z", 0.0, 50.0, D.z_um, 0.5,
                 help="Numune çıkış düzleminden dedektörle eşlenik düzleme mesafe (0 = odakta).")
        grid = st.selectbox("Grid size (N × N)", [64, 128, 256], index=[64, 128, 256].index(
            st.session_state.get("grid", D.grid)), key="grid")
        fov = st.selectbox("Field of view (µm)", [32.0, 48.0, 64.0, 96.0, 128.0], index=[32.0, 48.0, 64.0, 96.0, 128.0].index(
            st.session_state.get("fov", D.fov_um)), key="fov")
    with sb.expander("Hücre geometrisi ve malzeme", expanded=False):
        R = _num("Cell radius (µm)", "R", 5.0, 30.0, D.cell_radius_um, 0.5)
        Rn = _num("Nucleus radius (µm)", "Rn", 1.0, 15.0, D.nucleus_radius_um, 0.5)
        h = _num("Cell thickness h (µm)", "h", 0.5, 15.0, D.thickness_um, 0.5)
        mu_c = _num("μ_a cytoplasm (1/µm)", "muc", 0.0, 0.20, D.mu_cyto, 0.005, "%.3f")
        mu_n = _num("μ_a nucleus (1/µm)", "mun", 0.0, 0.50, D.mu_nuc, 0.005, "%.3f")
        n_c = _num("n cytoplasm", "nc", 1.335, 1.420, D.n_cyto, 0.001, "%.3f")
        n_n = _num("n nucleus", "nn", 1.335, 1.450, D.n_nuc, 0.001, "%.3f")
    with sb.expander("Foton bütçesi ve dedektör", expanded=True):
        lvl = st.select_slider("Photon budget N", options=list(PHOTON_LEVELS),
                               value=st.session_state.get("lvl", "Çok yüksek (10⁵)"), key="lvl")
        mode = st.radio("Budget definition", ["dose", "total"], horizontal=True,
                        index=["dose", "total"].index(st.session_state.get("mode", "dose")), key="mode",
                        help="dose: N foton (veya N çift) numuneye gelir. total: tüm kollarda üretilen toplam foton = N.")
        qe = _num("Detector efficiency η", "qe", 0.05, 1.0, D.qe, 0.05)
        bg = _num("Background (counts/px/acquisition)", "bg", 0.0, 200.0, D.background, 0.5)
        dk = _num("Dark counts (counts/px/acquisition)", "dk", 0.0, 50.0, D.dark, 0.5)
        reps = _num("Simulation repeats", "reps", 1, 10, D.repeats, 1)
        seed = st.number_input("Random seed", 0, 10**9, st.session_state.get("seed", D.seed), key="seed")
    cfg = SimConfig(wavelength_um=lam, NA=NA, z_um=z, grid=int(grid), fov_um=float(fov), cell_radius_um=R,
                    nucleus_radius_um=min(Rn, 0.7 * R * D.cell_aspect), thickness_um=h, mu_cyto=mu_c,
                    mu_nuc=mu_n, n_cyto=n_c, n_nuc=n_n, photons=PHOTON_LEVELS[lvl], budget_mode=mode, qe=qe,
                    background=bg, dark=dk, repeats=int(reps), seed=int(seed))
    for w in cfg.sampling_warnings():
        sb.warning(w)
    sb.caption(f"Δx = {cfg.dx:.3f} µm · detector pixel = {cfg.dx_det:.2f} µm · "
               f"{cfg.n_det}×{cfg.n_det} pixels · coherent cut-off {cfg.coherent_cutoff:.2f} c/µm")
    st.session_state["cfg"] = cfg
    return cfg


def cfg_key(cfg: SimConfig) -> tuple:
    return tuple(sorted(asdict(cfg).items()))


def cfg_from_key(key: tuple) -> SimConfig:
    return SimConfig(**dict(key))


def physics_key(cfg: SimConfig) -> tuple:
    """Only parameters that change the optics/cell (not photons/noise/seed) -> cache key for models."""
    return cfg_key(cfg.with_(photons=1.0, background=0.0, dark=0.0, repeats=1, seed=0, budget_mode="dose", qe=1.0))
