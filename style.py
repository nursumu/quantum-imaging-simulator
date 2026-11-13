"""Visual style and reusable UI helpers (navy / white / light-grey scientific theme)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

NAVY = "#13294B"
NAVY_2 = "#1F4E8C"
GREY_BG = "#F2F4F7"
INK = "#1E2430"
MUTED = "#5B6472"
# Fixed categorical order (never cycled): blue, orange, aqua, yellow, magenta
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SYMBOLS = ["circle", "square", "triangle-up", "diamond", "triangle-down"]

CSS = f"""
<style>
html, body, [class*="css"] {{ font-family: "Source Sans Pro", "Helvetica Neue", Arial, sans-serif; }}
header[data-testid="stHeader"] {{ background: {NAVY}; }}
header[data-testid="stHeader"] a, header[data-testid="stHeader"] span, header[data-testid="stHeader"] p,
header[data-testid="stHeader"] button {{ color: #FFFFFF !important; }}
header[data-testid="stHeader"] svg {{ fill: #FFFFFF; color: #FFFFFF; }}
section[data-testid="stSidebar"] {{ background: {GREY_BG}; border-right: 1px solid #DDE2EA; }}
.block-container {{ padding-top: 4.2rem; max-width: 1400px; }}
h1, h2, h3 {{ color: {NAVY}; }}
.qi-hero {{ background: {NAVY}; color: white; padding: 1.4rem 1.6rem; border-radius: 6px; margin-bottom: 1rem; }}
.qi-hero h1 {{ color: white; font-size: 1.75rem; margin: 0 0 .3rem 0; }}
.qi-hero p {{ color: #D6DEEA; margin: 0; font-size: 1.02rem; }}
.qi-panel {{ background: {GREY_BG}; border: 1px solid #E1E5EC; border-radius: 6px; padding: 1rem 1.1rem; margin-bottom: .8rem; }}
.qi-card {{ background: {GREY_BG}; border-left: 4px solid {NAVY_2}; border-radius: 4px; padding: .9rem 1rem; min-height: 235px; }}
.qi-card h4 {{ color: {NAVY}; margin-top: 0; }}
.qi-note {{ font-size: .88rem; color: {MUTED}; }}
.qi-tag {{ display:inline-block; background:#E3E9F2; color:{NAVY}; border-radius:3px; padding:1px 6px; font-size:.78rem; margin-right:4px; }}
div[data-testid="stMetric"] {{ background: {GREY_BG}; border: 1px solid #E1E5EC; border-radius: 6px; padding: .5rem .7rem; }}
</style>
"""


def apply_style():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str):
    st.markdown(f'<div class="qi-hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def panel(html: str):
    st.markdown(f'<div class="qi-panel">{html}</div>', unsafe_allow_html=True)


def note(text: str):
    st.markdown(f'<div class="qi-note">{text}</div>', unsafe_allow_html=True)


def heatmap(img, dx, title, colorscale="Gray", zmin=None, zmax=None, unit="", height=330, symmetric=False):
    img = np.asarray(img, dtype=float)
    n = img.shape[0]
    c = (np.arange(n) - n / 2 + 0.5) * dx
    if symmetric:
        m = np.nanmax(np.abs(img)) or 1.0
        zmin, zmax = -m, m
    fig = go.Figure(go.Heatmap(z=img, x=c, y=c, colorscale=colorscale, zmin=zmin, zmax=zmax,
                               colorbar=dict(title=dict(text=unit, side="right"), thickness=12),
                               hovertemplate="x=%{x:.1f} µm<br>y=%{y:.1f} µm<br>value=%{z:.4g}<extra></extra>"))
    fig.update_layout(title=dict(text=title, font=dict(size=13, color=NAVY)), height=height,
                      margin=dict(l=10, r=10, t=36, b=10), plot_bgcolor="white", paper_bgcolor="white")
    fig.update_xaxes(title="x (µm)", showgrid=False, constrain="domain")
    fig.update_yaxes(title="y (µm)", showgrid=False, scaleanchor="x", scaleratio=1, constrain="domain")
    return fig


def show_heatmaps(items, dx, cols=3, height=330):
    """items: list of dicts with keys img, title, (colorscale, zmin, zmax, unit, symmetric)."""
    for i in range(0, len(items), cols):
        cs = st.columns(cols)
        for c, it in zip(cs, items[i:i + cols]):
            with c:
                st.plotly_chart(heatmap(it["img"], it.get("dx", dx), it["title"], it.get("colorscale", "Gray"),
                                        it.get("zmin"), it.get("zmax"), it.get("unit", ""), height,
                                        it.get("symmetric", False)),
                                width="stretch", config={"displaylogo": False})


def line_chart(x, series: dict, errors: dict | None = None, title="", xlab="", ylab="", logx=True, logy=False,
               height=360, colors: dict | None = None):
    """colors: optional {series name: palette index} so a series keeps its colour across charts."""
    fig = go.Figure()
    for j, (name, y) in enumerate(series.items()):
        i = (colors or {}).get(name, j)
        e = None if errors is None else errors.get(name)
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines+markers",
                                 line=dict(color=SERIES[i % 5], width=2),
                                 marker=dict(symbol=SYMBOLS[i % 5], size=9, color=SERIES[i % 5],
                                             line=dict(color="white", width=1.5)),
                                 error_y=None if e is None else dict(type="data", array=e, thickness=1.2, width=4),
                                 hovertemplate=f"{name}<br>x=%{{x:.3g}}<br>y=%{{y:.4g}}<extra></extra>"))
    fig.update_layout(title=dict(text=title, font=dict(size=13, color=NAVY)), height=height,
                      margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white", paper_bgcolor="white",
                      legend=dict(orientation="h", yanchor="top", y=-0.22, font=dict(size=11)),
                      hovermode="x unified")
    fig.update_xaxes(title=xlab, type="log" if logx else "linear", gridcolor="#E6E9EE", zeroline=False)
    fig.update_yaxes(title=ylab, type="log" if logy else "linear", gridcolor="#E6E9EE", zeroline=False)
    return fig


def equation(title: str, latex: list[str] | str, body_md: str, expanded=False):
    with st.expander(title, expanded=expanded):
        for lx in ([latex] if isinstance(latex, str) else latex):
            st.latex(lx)
        st.markdown(body_md)


def fmt_n(x) -> str:
    """Compact number for st.metric tiles."""
    try:
        x = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(x):
        return "—"
    if abs(x) >= 1e5:
        m, e = f"{x:.2e}".split("e")
        return f"{float(m):.2f}e{int(e)}"
    if abs(x) >= 100 or x == int(x):
        return f"{x:,.0f}"
    return f"{x:.3g}"
