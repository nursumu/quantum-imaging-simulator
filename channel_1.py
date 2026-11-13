import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from calculations.metrics import detector_masks, reference_transmittance, image_metrics, summarize
from calculations.propagation import freq_grid, kz_map
from core.experiment import seed_for
from models.coherent_model import CoherentModel
from ui import cache
from ui.settings import physics_key
from ui.style import hero, show_heatmaps, equation, note, NAVY, SERIES, fmt_n

cfg = st.session_state.get("cfg") or __import__("ui.settings", fromlist=["x"]).global_settings()
hero("Kanal 1 · Classical coherent intensity imaging",
     "Klasik Koherent Işık Şiddeti Görüntüleme — shot-noise-limited baseline")

with st.container(border=True):
    st.markdown("**Kanala özgü kontroller** · ortak parametreler (N, λ, z, η, arka plan, kalınlık) sol paneldedir")
    k1, k2, k3, k4 = st.columns(4)
    A0 = k1.number_input("Incident amplitude A₀ (arb.)", 0.1, 10.0, 1.0, 0.1,
                         help="Foton normalizasyonu Σ|U_in|²'ye bölündüğü için A₀ sayımları değiştirmez (doğrulanır).")
    abs_scale = k2.slider("Absorption scale (explore)", 0.0, 3.0, 1.0, 0.1,
                          help="Yalnızca bu sayfada keşif için μ_a değerlerini ölçekler. Karşılaştırma sayfası global hücreyi kullanır.")
    ph_scale = k3.slider("Phase-delay scale (explore)", 0.0, 3.0, 1.0, 0.1,
                         help="Yalnızca bu sayfada Δn = n − n_medium değerlerini ölçekler.")
    bg_err = k4.slider("Assumed background / true", 0.5, 1.5, 1.0, 0.05,
                       help="Yeniden oluşturmada kullanılan arka plan tahmini. 1 = tam bilinen arka plan.")

nm = cfg.n_medium
lc = cfg.with_(A0=A0, mu_cyto=cfg.mu_cyto * abs_scale, mu_nuc=cfg.mu_nuc * abs_scale,
               mu_mem=cfg.mu_mem * abs_scale, mu_gran=cfg.mu_gran * abs_scale,
               n_cyto=nm + (cfg.n_cyto - nm) * ph_scale, n_nuc=nm + (cfg.n_nuc - nm) * ph_scale,
               n_mem=nm + (cfg.n_mem - nm) * ph_scale, n_gran=nm + (cfg.n_gran - nm) * ph_scale)
explore = abs_scale != 1.0 or ph_scale != 1.0
if explore:
    st.warning("Keşif modu: hücre global hücreden farklı (soğurma/faz ölçeklendi). Kanallar arası karşılaştırma için 1.0 kullanın.")

pk = physics_key(lc)
cell = cache.cell(pk)
model = CoherentModel(lc, cell)

runs, last = [], None
ref = reference_transmittance(cell["tau"], cfg.bin_factor)
masks = detector_masks(cell, cfg.bin_factor)
for r in range(cfg.repeats):
    rng = np.random.default_rng(seed_for(cfg.seed, "ch1page", r))
    out = model.sample(rng, bg_factor=bg_err)
    runs.append(image_metrics(out["images"]["coherent"], ref, model.sys["coherent"], masks))
    if last is None:
        last = out
ex = last["extras"]

st.markdown("#### Foton bütçesi (ilk tekrar)")
m = st.columns(6)
m[0].metric("Incident N_in", fmt_n(ex["N_incident"]))
m[1].metric("After sample", fmt_n(ex["N_after_sample"]), help="absorption losses")
m[2].metric("After pupil", fmt_n(ex["N_after_pupil"]), help="scattering outside NA")
m[3].metric("Exp. signal counts", fmt_n(ex["expected_signal_counts"]), help="η × photons after pupil")
m[4].metric("Exp. bg + dark", fmt_n(ex["expected_noise_counts"]))
m[5].metric("Detected (Poisson)", fmt_n(ex["detected_total"]))

dx, dxd = cfg.dx, cfg.dx_det
st.markdown("#### Optik alanlar")
tabs = st.tabs(["A–C · Incident field, t(x,y), exit field", "D · Angular spectrum & propagated field",
                "E–G · Intensity, expected & noisy counts", "Reconstruction"])
with tabs[0]:
    show_heatmaps([
        dict(img=np.abs(model.U_in), title="|U_in| = A₀ (uniform plane wave)", zmin=0, zmax=A0 * 1.05),
        dict(img=np.abs(cell["t"]), title="|t(x,y)| = exp(−α/2)", zmin=0, zmax=1),
        dict(img=np.angle(cell["t"]), title="arg t(x,y) (rad)", colorscale="Twilight", zmin=-np.pi, zmax=np.pi),
        dict(img=np.abs(model.U0), title="|U₀| = |U_in t|", zmin=0),
        dict(img=np.angle(model.U0), title="arg U₀ (rad)", colorscale="Twilight", zmin=-np.pi, zmax=np.pi),
        dict(img=np.abs(model.U0) ** 2, title="|U₀|² (exit-plane intensity)", zmin=0),
    ], dx)
with tabs[1]:
    FX, FY = freq_grid(cfg.grid, dx)
    spec = np.fft.fftshift(np.log10(np.abs(model.F_U0) ** 2 + 1e-12))
    f = np.fft.fftshift(np.fft.fftfreq(cfg.grid, dx))
    fig = go.Figure(go.Heatmap(z=spec, x=f, y=f, colorscale="Blues",
                               colorbar=dict(title="log₁₀|F(U₀)|²", thickness=12)))
    th = np.linspace(0, 2 * np.pi, 200)
    for rad, name, col in [(cfg.coherent_cutoff, "pupil NA/λ₀", SERIES[1]),
                           (1 / cfg.wavelength_medium, "evanescent limit 1/λ_m", SERIES[2])]:
        fig.add_trace(go.Scatter(x=rad * np.cos(th), y=rad * np.sin(th), mode="lines", name=name,
                                 line=dict(color=col, width=2)))
    fig.update_layout(title=dict(text="Angular spectrum of U₀ (f_x, f_y in cycles/µm)", font=dict(color=NAVY, size=13)),
                      height=420, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.15))
    fig.update_yaxes(scaleanchor="x")
    c1, c2 = st.columns([1, 1])
    c1.plotly_chart(fig, width="stretch")
    kz = kz_map(cfg.grid, dx, cfg.wavelength_medium)
    frac_ev = float((kz.imag > 0).mean())
    with c2:
        note(f"Grid Nyquist = {cfg.nyquist:.2f} c/µm · evanescent limit 1/λ_m = {1/cfg.wavelength_medium:.2f} c/µm · "
             f"pupil cut-off = {cfg.coherent_cutoff:.2f} c/µm. Gridde sönümlenen (evanescent) frekans oranı: "
             f"{100*frac_ev:.1f} %.")
        show_heatmaps([dict(img=np.abs(model.Uz), title=f"|U_z| after ASM (z = {cfg.z_um} µm) + pupil", zmin=0),
                       dict(img=np.angle(model.Uz), title="arg U_z (rad)", colorscale="Twilight", zmin=-np.pi, zmax=np.pi)],
                      dx, cols=2, height=300)
with tabs[2]:
    show_heatmaps([
        dict(img=model.I_det, title="E · Ideal detector intensity I = |U_z|² (pixel-binned)", zmin=0, dx=dxd),
        dict(img=last["mu_total"], title="F · Expected counts μ_total = μ_sig + b + d", colorscale="Magma", dx=dxd),
        dict(img=last["counts"], title="G · Noisy counts N ~ Poisson(μ_total)", colorscale="Magma", dx=dxd),
    ], dxd)
with tabs[3]:
    show_heatmaps([
        dict(img=ref, title="Reference T_ref = ⟨e^{-τ}⟩_pixel", zmin=0, zmax=1.2, dx=dxd),
        dict(img=model.I_det, title="System-limited T_sys (infinite photons)", zmin=0, zmax=1.2, dx=dxd),
        dict(img=last["images"]["coherent"], title="Reconstructed T̂ = (N − b̂)/(η N_in/N_pix)", zmin=0, zmax=1.2, dx=dxd),
    ], dxd)
    row = cfg.n_det // 2
    xs = (np.arange(cfg.n_det) - cfg.n_det / 2 + 0.5) * dxd
    fig = go.Figure()
    for i, (nme, y) in enumerate([("T_ref", ref[row]), ("T_sys", model.I_det[row]),
                                  ("T̂ (noisy)", last["images"]["coherent"][row])]):
        fig.add_trace(go.Scatter(x=xs, y=y, name=nme, mode="lines+markers", line=dict(color=SERIES[i], width=2),
                                 marker=dict(size=5)))
    fig.update_layout(title="Central row profile", height=320, xaxis_title="x (µm)", yaxis_title="transmittance",
                      margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white")
    st.plotly_chart(fig, width="stretch")

st.markdown(f"#### Görüntü kalitesi ({cfg.repeats} tekrar, ortalama ± std)")
s = summarize(runs)
st.dataframe(pd.DataFrame([{k: f"{s[k]:.3f} ± {s[k+'_std']:.3f}" for k in ["SNR_noise", "SNR_object", "CNR", "SSIM", "RMSE_object"]}]),
             hide_index=True, width="stretch")

st.markdown("#### Matematiksel açıklamalar")
equation("A · Coherent illumination", r"U_{\rm in}(x,y)=A_0",
         "Tek renkli (monochromatic), uzaysal olarak tam koherent düzlem dalga. |A₀|² birim alan başına şiddettir; "
         "foton sayısına dönüşümde Σ|U_in|²'ye bölündüğü için A₀'ın mutlak değeri sonucu değiştirmez.")
equation("B–C · Complex transmission and exit field", [r"t(x,y)=e^{-\alpha(x,y)/2}e^{i\phi(x,y)}",
                                                       r"U_0(x,y)=U_{\rm in}(x,y)\,t(x,y)"],
         "İnce numune yaklaşımı: numune alanı yalnızca yerel olarak çarpar. α boyutsuz optik derinliktir.")
equation("D · Angular Spectrum Method (ASM)",
         [r"U_z=\mathcal{F}^{-1}\!\left[\mathcal{F}(U_0)\,H(f_x,f_y,z)\,P(f_x,f_y)\right]",
          r"H=e^{ik_z z},\quad k_z=\sqrt{k^2-(2\pi f_x)^2-(2\pi f_y)^2},\quad k=\frac{2\pi n}{\lambda_0}"],
         "- **Propagating** bileşenler (k_x²+k_y² ≤ k²): k_z gerçel, |H| = 1, enerji korunur.\n"
         "- **Evanescent** bileşenler (k_x²+k_y² > k²): k_z = iκ, H = e^{−κz}; birkaç dalga boyu içinde söner ve dedektöre enerji taşımaz.\n"
         "- **P**: görüntüleme sisteminin koherent transfer fonksiyonu, kesim NA/λ₀. Bu dairenin dışına saçılan ışık *kaybolur* (yeniden normalize edilmez).\n"
         "- Sayısal yaklaşım: FFT periyodik sınır koşulu; uzun z için Matsushima bant sınırlaması uygulanır.")
equation("E · Detector intensity", r"I(x,y)=|U_z(x,y)|^2",
         "Dedektör pikseli 2×2 optik örneğin ortalamasıdır (şiddet bir yoğunluktur).")
equation("F · Expected photon counts",
         [r"\mu_{\rm sig}(i,j)=N_{\rm in}\,\eta\,\frac{|U_z(i,j)|^2}{\sum_{i,j}|U_{\rm in}(i,j)|^2}",
          r"\mu_{\rm total}(i,j)=\mu_{\rm sig}(i,j)+b+d"],
         "N_in: görüş alanına gelen foton sayısı; η: dedektör kuantum verimi; b: arka plan; d: karanlık sayım "
         "(piksel başına, tüm ölçüm süresi). Payda **gelen** güç olduğu için soğurma ve pupil kayıpları korunur.")
equation("G · Poisson statistics and reconstruction",
         [r"N(i,j)\sim{\rm Poisson}\big(\mu_{\rm total}(i,j)\big)",
          r"\hat T(i,j)=\frac{N(i,j)-\hat b-\hat d}{\eta N_{\rm in}/N_{\rm pix}}"],
         "Arka plan tam biliniyorsa T̂, I_det'in yansız (unbiased) tahminidir (Doğrulama sayfasında test edilir). "
         "SNR ≈ √(algılanan foton) shot-noise sınırıdır.")
with st.expander("Varsayımlar ve sınırlamalar"):
    st.markdown("- Skaler, aberasyonsuz, birim büyütmeli optik; ideal dairesel pupil.\n"
                "- İdeal foton sayan dedektör: okuma gürültüsü, ölü zaman, doygunluk yok.\n"
                "- Faz bilgisi yoğunluk ölçümünde kaybolur; z > 0'da faz, Fresnel kırınımı yoluyla kontrasta dönüşür.")
