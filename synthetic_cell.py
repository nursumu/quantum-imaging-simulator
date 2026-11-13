import numpy as np
import pandas as pd
import streamlit as st

from calculations.metrics import reference_transmittance
from ui import cache
from ui.settings import physics_key
from ui.style import hero, show_heatmaps, equation, note

cfg = st.session_state.get("cfg") or __import__("ui.settings", fromlist=["x"]).global_settings()
hero("Sentetik Hücre Oluşturucu · Synthetic cell generator",
     "Üç kanalın da kullandığı ortak hücre modeli — parametreler sol paneldeki ‘Hücre geometrisi ve malzeme’ bölümündedir")

pk = physics_key(cfg)
with st.spinner("Building synthetic cell ..."):
    cell = cache.cell(pk)

m = cell["masks"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Max optical depth τ", f"{cell['tau'].max():.3f}")
c2.metric("Min exp(−τ)", f"{np.exp(-cell['tau']).min():.3f}")
c3.metric("Max phase φ (rad)", f"{cell['phi'].max():.2f}")
c4.metric("Mean |t|² over FOV", f"{np.mean(np.abs(cell['t'])**2):.4f}")
c5.metric("Granules placed", f"{cell['n_granules']}")

dx = cfg.dx
tabs = st.tabs(["Geometri ve maske", "Soğurma ve optik derinlik", "Kırılma indisi ve faz", "Kompleks geçirgenlik t(x,y)"])
with tabs[0]:
    show_heatmaps([
        dict(img=cell["labels"], title="1 · Geometry: 1 membrane · 2 cytoplasm · 3 nucleus · 4 granule",
             colorscale=[[0, "#FFFFFF"], [0.25, "#eb6834"], [0.5, "#9EC5F4"], [0.75, "#13294B"], [1, "#eda100"]],
             zmin=0, zmax=4),
        dict(img=m["cell"].astype(float), title="2 · Cell mask", colorscale="Greys", zmin=0, zmax=1),
        dict(img=cell["L_total"], title="Cell thickness L(x,y)", colorscale="Greys", unit="µm"),
    ], dx)
with tabs[1]:
    show_heatmaps([
        dict(img=cell["mu_mean"], title="3 · Absorption: path-averaged μ_a (1/µm)", colorscale="Blues", unit="1/µm"),
        dict(img=cell["tau"], title="4 · Optical depth τ = Σ μ_a,j L_j (dimensionless)", colorscale="Blues", unit="τ"),
        dict(img=np.exp(-cell["tau"]), title="Intensity transmittance exp(−τ) (Beer–Lambert)", zmin=0, zmax=1),
    ], dx)
with tabs[2]:
    show_heatmaps([
        dict(img=cell["n_mean"], title="5 · Path-averaged refractive index n̄", colorscale="Oranges", unit="n"),
        dict(img=cell["opd"], title="Optical path difference Σ Δn_j L_j (µm)", colorscale="Oranges", unit="µm"),
        dict(img=cell["phi"], title="6 · Phase φ = k₀ Σ Δn_j L_j (rad)", colorscale="Oranges", unit="rad"),
    ], dx)
with tabs[3]:
    show_heatmaps([
        dict(img=np.abs(cell["t"]), title="7 · |t(x,y)| = exp(−τ/2)", zmin=None, zmax=1),
        dict(img=np.angle(cell["t"]), title="8 · arg t(x,y) (wrapped, rad)", colorscale="Twilight",
             zmin=-np.pi, zmax=np.pi, unit="rad"),
    ], dx, cols=2, height=380)

st.markdown("#### Kompartman özellikleri (aktif değerler)")
nm = cfg.n_medium
df = pd.DataFrame({
    "Compartment": ["Cytoplasm", "Nucleus", "Membrane", "Granules", "Medium"],
    "μ_a (1/µm)": [cfg.mu_cyto, cfg.mu_nuc, cfg.mu_mem, cfg.mu_gran, 0.0],
    "n": [cfg.n_cyto, cfg.n_nuc, cfg.n_mem, cfg.n_gran, nm],
    "Δn = n − n_medium": [cfg.n_cyto - nm, cfg.n_nuc - nm, cfg.n_mem - nm, cfg.n_gran - nm, 0.0],
    "Pixels in mask": [int(m["cytoplasm"].sum()), int(m["nucleus"].sum()), int(m["membrane"].sum()),
                       int((cell["L_gran"] > 0).sum()), int(m["background"].sum())],
})
st.dataframe(df, hide_index=True, width="stretch")
note("Sitoplazma μ_a ve Δn değerleri ±%35 düzgün (smooth) rastgele heterojenlik ile çarpılır (sabit tohum → "
     "tekrarlanabilir). Zar ve granül değerleri sabittir. Değerler boyanmış bir hücreye benzer varsayımlardır; ölçüm değildir.")

equation("Matematiksel model — optik derinlik, faz ve kompleks geçirgenlik",
         [r"\tau(x,y)=\sum_j \mu_{a,j}\,L_j(x,y)\qquad T(x,y)=e^{-\tau(x,y)}",
          r"\phi(x,y)=k_0\sum_j \big(n_j-n_{\rm medium}\big)\,L_j(x,y),\qquad k_0=\frac{2\pi}{\lambda_0}",
          r"t(x,y)=e^{-\alpha(x,y)/2}\,e^{i\phi(x,y)},\qquad \alpha\equiv\tau"],
         "- **μ_a,j** [µm⁻¹]: absorption coefficient of compartment *j* (malzeme özelliği).\n"
         "- **L_j(x,y)** [µm]: geometrik yol uzunluğu (projection approximation).\n"
         "- **τ = α** (boyutsuz): *optical depth*. Absorption coefficient ile karıştırılmamalıdır: τ, μ_a'nın yol boyunca integralidir.\n"
         "- Genlik zayıflaması exp(−τ/2)'dir çünkü şiddet |t|² = exp(−τ) olmalıdır (Beer–Lambert).\n"
         "- **φ** [rad]: ortama göre ek faz; kırılma indisi farkı × yol uzunluğu × k₀.\n"
         "- Geometri: hücre bir yarı-elipsoid kubbe (yükseklik h); zar, kalınlığı d_m olan bir kabuk; çekirdek "
         "yüksekliği 0.7h olan elipsoid.", expanded=True)
with st.expander("Varsayımlar ve sınırlamalar"):
    st.markdown(
        "- **Thin-object (projection) approximation:** hücre içinde kırınım/çoklu saçılma yok. Geçerlilik: "
        f"h ≪ alan derinliği ≈ λ/NA² = {cfg.wavelength_um / cfg.NA**2:.1f} µm. Daha kalın hücrelerde yaklaşık.\n"
        "- Skaler optik; polarizasyon yok.\n"
        "- Optik sabitler literatür aralığında varsayımlardır, deneysel olarak uydurulmamıştır.\n"
        "- Aynı `cell_seed` ile hücre her zaman aynıdır; üç kanal da bu hücreyi ve aynı referans görüntüyü "
        "(T_ref = piksel ortalamalı exp(−τ)) kullanır.")
ref = reference_transmittance(cell["tau"], cfg.bin_factor)
st.markdown("#### Ortak referans görüntü (dedektör ızgarasında)")
show_heatmaps([dict(img=ref, title=f"T_ref on {cfg.n_det}×{cfg.n_det} detector grid", zmin=0, zmax=1.05,
                    dx=cfg.dx_det)], cfg.dx_det, cols=2)
