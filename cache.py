"""Cached model construction (expensive physics computed once per configuration).

Models depend only on the optics/cell parameters (physics_key); photon number,
efficiency, background and seed are applied at sampling time, so changing them
does not rebuild the models.
"""
from __future__ import annotations

import streamlit as st

from core.experiment import ModelSet
from models.cell_model import build_cell
from models.classical_correlation_model import ClassicalCorrelationModel
from models.coherent_model import CoherentModel
from models.entangled_photon_model import EntangledPhotonModel
from ui.settings import cfg_from_key


@st.cache_resource(show_spinner=False, max_entries=8)
def cell(pkey):
    return build_cell(cfg_from_key(pkey))


@st.cache_resource(show_spinner=False, max_entries=8)
def ch1_model(pkey):
    c = cfg_from_key(pkey)
    return CoherentModel(c, cell(pkey))


@st.cache_resource(show_spinner=False, max_entries=8)
def ch2_model(pkey, frames, corr, speckle):
    c = cfg_from_key(pkey)
    return ClassicalCorrelationModel(c, cell(pkey), frames=frames, corr_strength=corr, speckle_um=speckle)


@st.cache_resource(show_spinner=False, max_entries=6)
def ch3_model(pkey, sigma_D, sigma_S, mu, r, state):
    c = cfg_from_key(pkey)
    return EntangledPhotonModel(c, cell(pkey), sigma_D=sigma_D, sigma_S=sigma_S, mu=mu, window_r=r, state=state)


@st.cache_resource(show_spinner=False, max_entries=4)
def model_set(pkey, ch2_opts: tuple, ch3_opts: tuple):
    return ModelSet(cfg_from_key(pkey), dict(ch2_opts), dict(ch3_opts))


def with_runtime(model, cfg):
    """Return the cached model bound to the CURRENT runtime parameters (photons, qe, noise, seed)."""
    model.cfg = cfg
    return model


def bind_set(ms, cfg):
    """Bind a cached ModelSet to the current runtime parameters."""
    ms.cfg = cfg
    for m in list(ms.models.values()) + list(ms.edge_models.values()):
        m.cfg = cfg
    return ms
