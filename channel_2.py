import numpy as np
import pandas as pd
import streamlit as st

from calculations.metrics import detector_masks, reference_transmittance, image_metrics, summarize
from core.experiment import seed_for
from ui import cache
from ui.cache import with_runtime
from ui.settings import physics_key
from ui.style import hero, show_heatmaps, equation, note, line_chart, fmt_n

cfg = st.session_state.get("cfg") or __import__("ui.settings", fromlist=["x"]).global_settings()
hero("Kanal 2 · Classically correlated imaging",
     "Klasik Korelasyonlu Işıkla Görüntüleme — split pseudo-thermal (speckle) light, NO entanglement")

with st.container(border=True):
    st.markdown("**Kanala özgü kontroller**")
    k1, k2, k3, k4 = st.columns(4)
    M = k1.select_slider("Measurement frames M", [10, 20, 50, 100, 200, 500, 1000], value=100)
    c = k2.slider("Correlation strength c", 0.0, 1.0, 1.0, 0.05,
                  help="E_r = √c E + √(1−c) E′ ; c = 0 → bağımsız (korelasyonsuz) ışınlar kontrolü")
    sp = k3.slider("Speckle correlation length ℓ_c (µm)", 1.0, 6.0, 2.5, 0.5)
    control = k4.toggle("Uncorrelated-beams control (c = 0)", value=False)
    if control:
        c = 0.0

pk = physics_key(cfg)
with st.spinner("Computing exact ensemble moments (Gaussian moment theorem) ..."):
    base = cache.ch2_model(pk, 100, 1.0, sp)
model = with_runtime(base.variant(corr_strength=c, frames=M), cfg)
cell = cache.cell(pk)
ref = reference_transmittance(cell["tau"], cfg.bin_factor)
masks = detector_masks(cell, cfg.bin_factor)

run = st.button("▶ Simülasyonu çalıştır", type="primary")
key = ("ch2", pk, M, c, sp, cfg.photons, cfg.qe, cfg.background, cfg.dark, cfg.seed, cfg.repeats, cfg.budget_mode)
if run or st.session_state.get("ch2_key") != key and "ch2_res" not in st.session_state:
    prog = st.progress(0.0, "Generating speckle frames ...")
    runs, first = [], None
    for r in range(cfg.repeats):
        rng = np.random.default_rng(seed_for(cfg.seed, "ch2page", r))
        out = model.sample(rng, keep=(r == 0))
        runs.append({k: image_metrics(v, ref, model.sys[k], masks) for k, v in out["images"].items()})
        first = first or out
        prog.progress((r + 1) / cfg.repeats, f"repeat {r+1}/{cfg.repeats}")
    prog.empty()
    st.session_state["ch2_res"], st.session_state["ch2_key"] = (runs, first), key
if "ch2_res" not in st.session_state:
    st.stop()
if st.session_state.get("ch2_key") != key:
    st.info("Parametreler değişti — güncel sonuçlar için ‘Simülasyonu çalıştır’a basın. Aşağıda önceki sonuç gösteriliyor.")
runs, out = st.session_state["ch2_res"]
ex, stt, f0 = out["extras"], out["stats"], out["frame0"]

st.markdown("#### Foton bütçesi")
m = st.columns(6)
m[0].metric("On sample (signal)", fmt_n(ex["N_incident"]))
m[1].metric("Reference arm", fmt_n(ex["N_reference"]))
m[2].metric("Total generated", fmt_n(ex["N_generated_total"]))
m[3].metric("Det. signal", fmt_n(ex["detected_signal"]))
m[4].metric("Det. reference", fmt_n(ex["detected_reference"]))
m[5].metric("Frames M", fmt_n(ex["frames"]))
m2 = st.columns(4)
m2[0].metric("Correlation c", f"{model.c:.2f}")
m2[1].metric("Pooled g²_sr", f"{ex['g2_sr_pooled']:.3f}", help="Klasik sınır: g²_sr ≤ 2")
m2[2].metric("Frame corr. coeff.", f"{ex['corr_coeff']:.3f}")
m2[3].metric("Median rel. unc. C_sr", "undefined" if not np.isfinite(ex["cov_rel_uncertainty"])
             else f"{ex['cov_rel_uncertainty']:.2f}")
if not ex["cov_estimator_defined"]:
    st.warning("Shot-noise-corrected reference variance D ≤ 0: too few photons per frame — the covariance estimator is undefined.")

dx, dxd = cfg.dx, cfg.dx_det
tabs = st.tabs(["1–2 · Source & beams", "3–6 · Detectors (one frame)", "7 · Cross-correlation", "8 · Reconstruction",
                "Parametre etkileri"])
with tabs[0]:
    show_heatmaps([
        dict(img=np.abs(f0["E_source"]) ** 2, title="Source speckle |E_f|² (one frame)", colorscale="Magma"),
        dict(img=np.abs(f0["E_ref_source"]) ** 2, title="Reference-arm source field |E_r,f|²", colorscale="Magma"),
        dict(img=f0["I_s"], title="Signal beam on detector I_s (noise-free)", colorscale="Magma", dx=dxd),
        dict(img=f0["I_r"], title="Reference beam on detector I_r (noise-free)", colorscale="Magma", dx=dxd),
    ], dx, cols=2, height=360)
with tabs[1]:
    show_heatmaps([
        dict(img=f0["n_s"], title="Signal detector counts (frame 1)", colorscale="Magma"),
        dict(img=f0["n_r"], title="Reference detector counts (frame 1)", colorscale="Magma"),
        dict(img=stt["mean_s"], title="⟨n_s⟩ over frames", colorscale="Magma"),
        dict(img=stt["mean_r"], title="⟨n_r⟩ over frames", colorscale="Magma"),
    ], dxd, cols=2, height=360)
with tabs[2]:
    show_heatmaps([
        dict(img=stt["cov"], title="C_sr = ⟨Δn_s Δn_r⟩ (counts²)", colorscale="Viridis"),
        dict(img=model.sys["cov"], title="Exact expectation (Gaussian moment theorem), T units", colorscale="Viridis"),
        dict(img=stt["g2"], title="g²_sr(X) = ⟨n_s n_r⟩/(⟨n_s⟩⟨n_r⟩)", colorscale="Viridis", zmin=0, zmax=3),
        dict(img=out["se_cov_T"], title="Uncertainty SE[T_cov] (T units)", colorscale="Viridis"),
    ], dxd, cols=2, height=360)
with tabs[3]:
    show_heatmaps([
        dict(img=ref, title="Reference T_ref", zmin=0, zmax=1.2),
        dict(img=out["images"]["conv"], title="Signal-arm image ⟨n_s⟩ (conventional)", zmin=0, zmax=1.2),
        dict(img=out["images"]["ratio"], title="Reference-normalised ⟨n_s⟩/⟨n_r⟩", zmin=0, zmax=1.2),
        dict(img=out["images"]["cov"], title="Correlation image (a_r/a_s)·C_sr/D", zmin=0, zmax=1.2),
    ], dxd, cols=2, height=360)
    rows = []
    for k, lab in [("conv", "signal arm (1st order)"), ("ratio", "reference-normalised"), ("cov", "covariance C_sr")]:
        s = summarize([r[k] for r in runs])
        rows.append({"Estimator": lab, **{q: f"{s[q]:.3f} ± {s[q+'_std']:.3f}" for q in ["SNR_noise", "SNR_object", "CNR", "SSIM"]}})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
with tabs[4]:
    note("Etkiler, mevcut ayarlarla örnekleme yapılarak hesaplanır (model önbellekte). Her nokta tek tekrar.")
    which = st.selectbox("Study", ["Correlation strength c", "Frames M (fixed photon budget)", "Photon number N",
                                   "Background b"])
    if st.button("Çalıştır (study)"):
        prog = st.progress(0.0)
        if which.startswith("Correlation"):
            xs = [0.0, 0.25, 0.5, 0.75, 1.0]
            mk = lambda v: (base.variant(corr_strength=v, frames=M), {})
            xl, logx = "correlation strength c", False
        elif which.startswith("Frames"):
            xs = [10, 30, 100, 300, 1000]
            mk = lambda v: (base.variant(corr_strength=c, frames=v), {})
            xl, logx = "frames M", True
        elif which.startswith("Photon"):
            xs = [1e2, 1e3, 1e4, 1e5, 1e6]
            mk = lambda v: (base.variant(corr_strength=c, frames=M), {"N": v})
            xl, logx = "photon budget N", True
        else:
            xs = [0.1, 1, 10, 100]
            mk = lambda v: (base.variant(corr_strength=c, frames=M), {"noise": v})
            xl, logx = "background b (counts/px)", True
        res = {"conv": [], "ratio": [], "cov": []}
        mean_cov = []
        for i, v in enumerate(xs):
            mm, kw = mk(v)
            mm = with_runtime(mm, cfg)
            o = mm.sample(np.random.default_rng(seed_for(cfg.seed, "study", which, v)), **kw)
            for k in res:
                res[k].append(image_metrics(o["images"][k], ref, mm.sys[k], masks)["SSIM"])
            mean_cov.append(float(np.nanmean(o["images"]["cov"][masks["background"]])))
            prog.progress((i + 1) / len(xs))
        prog.empty()
        a, b = st.columns(2)
        a.plotly_chart(line_chart(xs, {"signal arm": res["conv"], "ref-normalised": res["ratio"], "covariance": res["cov"]},
                                  title=f"SSIM vs {xl}", xlab=xl, ylab="SSIM", logx=logx), width="stretch")
        b.plotly_chart(line_chart(xs, {"⟨T_cov⟩ outside cell (expected = c)": mean_cov}, title="Correlation signal level",
                                  xlab=xl, ylab="⟨T_cov⟩", logx=logx), width="stretch")

st.markdown("#### Matematiksel açıklamalar")
equation("1 · Classical light source — why split pseudo-thermal light?",
         [r"E_f(\mathbf{x})=\mathcal{F}^{-1}\!\left[F(\mathbf{f})W_f(\mathbf{f})\right],\quad |F|^2\propto e^{-2\pi^2\ell_c^2 f^2},\quad \langle|E|^2\rangle=1"],
         "Dönen bir difüzörden geçen lazer her ölçüm karesinde bağımsız, tam gelişmiş bir speckle alanı üretir. Bu alan "
         "dairesel kompleks Gauss (termal) istatistiğe sahiptir: üstel şiddet dağılımı ve g²(0) = 2 (photon bunching). "
         "Bu, iki-foton görüntülemenin standart klasik analoğudur (Gatti 2004, Valencia 2005, Ferri 2005). Korelasyon fiziksel "
         "olarak ortak bir klasik dalgadan gelir; rastgele ilişkili diziler üretilmez.")
equation("2–4 · Beam splitting, sample interaction, propagation",
         [r"U_s=\mathcal{F}^{-1}\{HP\,\mathcal{F}[E_f\,t]\},\qquad U_r=\mathcal{F}^{-1}\{HP\,\mathcal{F}[E_{r,f}]\}",
          r"E_{r,f}=\sqrt{c}\,E_f+\sqrt{1-c}\,E'_f"],
         "Sinyal kolu hücreden geçer; referans kolu geçmez ve özdeş optikle görüntülenir. c korelasyon gücüdür; E′ bağımsız bir speckle'dır.")
equation("5–6 · Detection and photon statistics",
         [r"n_a(X,f)\sim{\rm Poisson}\!\left(\eta\,a_a\,I_a(X,f)+\frac{b+d}{M}\right),\quad a_a=\frac{N_a}{M\,N_{\rm pix}}"],
         "N_a: o koldaki foton sayısı (bütçe modu ile belirlenir). Arka plan aynı toplam ölçüm süresine yayılır.")
equation("7 · Cross-intensity correlation",
         [r"C_{sr}(x,y)=\langle\Delta I_s(x,y)\,\Delta I_r(x,y)\rangle,\qquad \Delta I=I-\langle I\rangle",
          r"\hat C_{sr}=\frac{1}{M-1}\sum_{f=1}^{M}\Delta n_s\,\Delta n_r,\qquad {\rm SE}=\frac{{\rm std}_f(\Delta n_s\Delta n_r)}{\sqrt M}",
          r"\mathbb{E}[\hat C_{sr}]=\eta^2 a_s a_r\,{\rm Cov}(I_s,I_r)=\eta^2 a_s a_r\,|\langle U_s U_r^*\rangle|^2"],
         "- ⟨·⟩: ölçüm kareleri (M) üzerinden ortalama; **ΔI**: kareden kareye şiddet dalgalanması.\n"
         "- C_sr, iki kolun dalgalanmalarının birlikte değişimidir. Hücre sinyal kolundaki speckle'ı |t|² ile ölçeklediği için C_sr ∝ T(x,y).\n"
         "- İki dedektörün Poisson gürültüleri bağımsızdır → kovaryansa yanlılık katmaz; ilişkisiz arka plan da ortalamada sıfırlanır (yalnızca varyans ekler).\n"
         "- Son eşitlik Gauss moment teoremidir; uygulama bunu tam olarak hesaplar (sistem-sınırlı görüntü).")
equation("8 · Reconstruction & alternative estimators",
         [r"\hat T_{\rm cov}=\frac{a_r}{a_s}\,\frac{\hat C_{sr}}{D},\qquad D=\overline{{\rm Var}_f(n_r)-\langle n_r\rangle}",
          r"\hat T_{\rm ratio}=\frac{(\langle n_s\rangle-\beta)/\eta a_s}{(\langle n_r\rangle-\beta)/\eta a_r}"],
         "D, referansın shot-noise düzeltilmiş varyansının uzaysal ortalamasıdır; arka plan bilgisi gerektirmez. Oran tahmincisi "
         "birinci derece bir referans normalizasyonudur ve karşılaştırma için gösterilir. Hangi tahmincinin daha iyi olduğu "
         "**varsayılmaz**; sonuçlardan okunur.")
with st.expander("Classical correlation ≠ quantum entanglement"):
    st.markdown("Bu kanaldaki korelasyonlar tamamen klasiktir: iki kol aynı rastgele klasik alanın kopyalarını taşır. "
                "Normalize çapraz korelasyon Cauchy–Schwarz eşitsizliğine uyar (g²_sr ≤ 2). Hiçbir dolanıklık iddiası yoktur.")
with st.expander("Varsayımlar ve sınırlamalar"):
    st.markdown("- İdeal, tam gelişmiş Gauss speckle; kareler istatistiksel olarak bağımsız.\n"
                "- İki kol mükemmel hizalı ve özdeş optiğe sahip.\n"
                "- Sonlu M ile kovaryans tahmini gürültülüdür; hata ∝ M^{-1/2} (Doğrulama sayfası).")
