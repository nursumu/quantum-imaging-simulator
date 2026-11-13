import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from calculations.metrics import detector_masks, reference_transmittance, image_metrics, summarize
from core.experiment import seed_for
from models.entangled_photon_model import (EntangledPhotonModel, epr_analytic, measured_epr_product,
                                           biphoton_density, biphoton_density_k, joint_row_matrix)
from ui import cache
from ui.cache import with_runtime
from ui.settings import physics_key
from ui.style import hero, show_heatmaps, equation, note, line_chart, fmt_n, NAVY

cfg = st.session_state.get("cfg") or __import__("ui.settings", fromlist=["x"]).global_settings()
hero("Kanal 3 · Entangled-photon quantum imaging",
     "Dolanık Fotonlarla Kuantum Görüntüleme — SPDC biphoton state, coincidence detection, EPR witness")

with st.container(border=True):
    st.markdown("**Kanala özgü kontroller**")
    k1, k2, k3, k4, k5 = st.columns(5)
    sD = k1.slider("σ_D: position correlation (µm)", 0.4, 3.0, 0.8, 0.1,
                   help="rms of x_s − x_i (phase-matching width, sample-plane units)")
    sS = k2.select_slider("σ_S: pump width (µm)", [50, 100, 200, 400, 800], value=400)
    mu = k3.select_slider("Pairs per coincidence window μ", [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0], value=0.1,
                          help="μ = N_pairs/M. Küçük μ = iyi zaman çözünürlüğü, az tesadüfi koinsidans.")
    r = k4.select_slider("Spatial coincidence window", [0, 1, 2], value=1, format_func=lambda v: f"{2*v+1}×{2*v+1} px")
    state = k5.radio("Source state", ["entangled", "separable"], format_func=lambda s: {
        "entangled": "Entangled (SPDC)", "separable": "Separable control"}[s])

pk = physics_key(cfg)
an = epr_analytic(max(sD, 0.75 * cfg.dx), sS)
cell = cache.cell(pk)
ref = reference_transmittance(cell["tau"], cfg.bin_factor)
masks = detector_masks(cell, cfg.bin_factor)

st.markdown("#### Biphoton state ve dolanıklık tanığı (entanglement witness)")
c1, c2, c3 = st.columns([1, 1, 0.9])
x = np.linspace(-6, 6, 241)
XS, XI = np.meshgrid(x, x)
kk = np.linspace(-max(4 / sS, 2.5 / sD) if state == "separable" else -max(6 / sS, 0.02),
                 max(4 / sS, 2.5 / sD) if state == "separable" else max(6 / sS, 0.02), 241)
KS, KI = np.meshgrid(kk, kk)
for col, Z, xx, lab, t in [(c1, biphoton_density(XS, XI, sD, sS, state), x, "µm", "|Ψ(x_s,x_i)|² near field"),
                           (c2, biphoton_density_k(KS, KI, sD, sS, state), kk, "rad/µm", "|Ψ̃(k_s,k_i)|² far field")]:
    fig = go.Figure(go.Heatmap(z=Z, x=xx, y=xx, colorscale="Blues", showscale=False))
    fig.update_layout(title=dict(text=f"{t} · {state}", font=dict(color=NAVY, size=13)), height=330,
                      margin=dict(l=10, r=10, t=40, b=10), xaxis_title=f"idler ({lab})", yaxis_title=f"signal ({lab})")
    fig.update_yaxes(scaleanchor="x")
    col.plotly_chart(fig, width="stretch")
with c3:
    prod, ci = measured_epr_product(np.random.default_rng(cfg.seed), 20000, state, sD, sS, dx_near=cfg.dx_det)
    st.metric("Schmidt number K (per axis, entangled)", f"{an['schmidt_per_axis']:.0f}")
    st.metric("EPR product, analytic (entangled)", f"{an['product']:.2e}")
    st.metric(f"EPR product, simulated ({state})", f"{prod:.2e}", help=f"95 % CI {ci[0]:.2e} – {ci[1]:.2e}")
    if ci[1] < 0.25:
        st.success("Δ²(x_s−x_i)·Δ²(k_s+k_i) < 1/4 → EPR steering ⇒ entanglement certified (for this model state).")
    else:
        st.warning("Product ≥ 1/4 → no entanglement certified (separable control behaves as expected).")

run = st.button("▶ Simülasyonu çalıştır", type="primary")
key = ("ch3", pk, sD, sS, mu, r, state, cfg.photons, cfg.qe, cfg.background, cfg.dark, cfg.seed, cfg.repeats, cfg.budget_mode)
if run or (st.session_state.get("ch3_key") != key and "ch3_res" not in st.session_state):
    prog = st.progress(0.0, "Computing conditional joint detection probabilities P(X_s | x_i) ...")
    model = cache.ch3_model(pk, float(sD), float(sS), float(mu), int(r), state)
    model = with_runtime(model, cfg)
    runs, first = [], None
    for i in range(cfg.repeats):
        prog.progress((i + 0.5) / cfg.repeats, f"event Monte Carlo, repeat {i+1}/{cfg.repeats}")
        o = model.sample(np.random.default_rng(seed_for(cfg.seed, "ch3page", i)), keep=(i == 0))
        runs.append({k: image_metrics(v, ref, model.sys[k], masks) for k, v in o["images"].items()})
        first = first or o
    prog.empty()
    st.session_state["ch3_res"], st.session_state["ch3_key"] = (runs, first, model.sys, model.signal_transmission), key
if "ch3_res" not in st.session_state:
    st.stop()
if st.session_state.get("ch3_key") != key:
    st.info("Parametreler değişti — güncel sonuçlar için ‘Simülasyonu çalıştır’a basın. Aşağıda önceki sonuç gösteriliyor.")
runs, out, sysimg, strans = st.session_state["ch3_res"]
ex, mp = out["extras"], out["maps"]

st.markdown("#### Foton çifti bütçesi (ilk tekrar)")
m = st.columns(6)
m[0].metric("Generated pairs", fmt_n(ex["pairs_generated"]))
m[1].metric("Signal lost", fmt_n(ex["signal_lost_sample_or_pupil"]))
m[2].metric("Det. signal", fmt_n(ex["signal_detected"]))
m[3].metric("Det. idler", fmt_n(ex["idler_detected"]))
m[4].metric("True coinc.", fmt_n(ex["coincidences_true"]), help=f"expected {ex['expected_true']:.0f}")
m[5].metric("Accidental coinc.", fmt_n(ex["coincidences_accidental"]),
            help=f"delayed-window estimate {ex['accidentals_delayed']:.0f}")
m2 = st.columns(6)
m2[0].metric("Windows M", fmt_n(ex["windows"]))
m2[1].metric("Noise s / i", f"{ex['signal_noise_counts']}/{ex['idler_noise_counts']}")
m2[2].metric("CAR", fmt_n(ex["CAR"]))
m2[3].metric("g²_si (whole sensor)", fmt_n(ex["g2_si"]))
m2[4].metric("g²_ss · g²_ii", f"{ex['g2_ss']*ex['g2_ii']:.3f}",
             help="Cauchy–Schwarz: klasik ışık için g²_si² ≤ g²_ss g²_ii")
m2[5].metric("Median rel. unc.", f"{ex['coinc_rel_uncertainty']:.2f}")
if ex["g2_si"] ** 2 > ex["g2_ss"] * ex["g2_ii"]:
    note("g²_si² > g²_ss·g²_ii: Cauchy–Schwarz eşitsizliği ihlal edilir → foton sayısı (zaman) korelasyonu klasik değildir. "
         "Bu tek başına <b>uzaysal dolanıklığı</b> kanıtlamaz; onun için yukarıdaki EPR tanığı kullanılır.")

dxd = cfg.dx_det
tabs = st.tabs(["Detection maps", "Coincidences, accidentals, G² and g²", "Reconstruction", "Joint detection P(x_s,x_i)",
                "Parametre etkileri"])
with tabs[0]:
    show_heatmaps([dict(img=mp["S_s"], title="Signal-photon detection map (singles)", colorscale="Magma"),
                   dict(img=mp["S_i"], title="Idler-photon detection map (singles)", colorscale="Magma"),
                   dict(img=out["images"]["singles"], title="Signal singles image (no heralding)", zmin=0, zmax=1.4)],
                  dxd)
with tabs[1]:
    with np.errstate(divide="ignore", invalid="ignore"):
        car_map = np.where(mp["A"] > 0, (mp["C"] - mp["A"]) / mp["A"], np.nan)
    show_heatmaps([dict(img=mp["C"], title="Coincidence counts C(X)", colorscale="Magma"),
                   dict(img=mp["A"], title="Accidental coincidences (delayed window)", colorscale="Magma"),
                   dict(img=car_map, title="Coincidence / accidental ratio (C−A)/A per pixel", colorscale="Viridis"),
                   dict(img=mp["G2"], title="G²(X) = ⟨n_s n_i⟩ per window = C/M", colorscale="Viridis"),
                   dict(img=mp["g2"], title="g²(X) = M·C / (S_s·S_i(W))", colorscale="Viridis"),
                   dict(img=mp["unc"], title="Uncertainty of T̂ (Poisson, incl. accidental subtraction)", colorscale="Viridis")],
                  dxd)
with tabs[2]:
    show_heatmaps([dict(img=ref, title="Reference T_ref", zmin=0, zmax=1.2),
                   dict(img=sysimg["coincidence"], title="System-limited coincidence image", zmin=0, zmax=1.2),
                   dict(img=out["images"]["coincidence"], title="Reconstructed T̂ = (C − A)/(η_s η_i N K_flat)",
                        zmin=0, zmax=1.2)], dxd)
    rows = []
    for k, lab in [("coincidence", "coincidence image"), ("singles", "signal singles (no coincidence)")]:
        s = summarize([q[k] for q in runs])
        rows.append({"Estimator": lab, **{q: f"{s[q]:.3f} ± {s[q+'_std']:.3f}" for q in ["SNR_noise", "SNR_object", "CNR", "SSIM"]}})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
with tabs[3]:
    sig, idl = out["events"]
    J = joint_row_matrix(sig, idl, cfg.n_det, cfg.n_det // 2)
    c = (np.arange(cfg.n_det) - cfg.n_det / 2 + 0.5) * dxd
    fig = go.Figure(go.Heatmap(z=J, x=c, y=c, colorscale="Magma", colorbar=dict(title="counts")))
    fig.update_layout(title="Measured joint coincidences along the central row (all same-window pairs, no spatial window)",
                      height=420, xaxis_title="idler x (µm)", yaxis_title="signal x (µm)", margin=dict(l=10, r=10, t=40, b=10))
    fig.update_yaxes(scaleanchor="x")
    a, b = st.columns([1.2, 1])
    a.plotly_chart(fig, width="stretch")
    with b:
        note("Köşegen = gerçek koinsidanslar (x_s ≈ x_i, near-field konum korelasyonu). Köşegen dışı = tesadüfi koinsidanslar "
             "(farklı çiftler veya arka plan aynı pencereye düşer). Ortak algılama olasılığı P(X_s,x_i), her üretilen çift "
             f"için modelde tam olarak hesaplanır; çift başına ortalama sinyal iletimi = {strans:.3f}.")
with tabs[4]:
    note("Mevcut model ile örnekleme (tek tekrar). μ ve arka plan, tesadüfi koinsidansları kontrol eder.")
    which = st.selectbox("Study", ["Pairs per window μ (timing)", "Background b", "Photon-pair number N"])
    if st.button("Çalıştır (study)"):
        model = with_runtime(cache.ch3_model(pk, float(sD), float(sS), float(mu), int(r), state), cfg)
        prog = st.progress(0.0)
        if which.startswith("Pairs"):
            xs, kwf, xl = [0.01, 0.1, 1.0, 10.0], (lambda v: {"mu": v}), "pairs per window μ"
        elif which.startswith("Background"):
            xs, kwf, xl = [0.1, 1, 10, 100, 1000], (lambda v: {"noise": v}), "background b (counts/px)"
        else:
            xs, kwf, xl = [1e2, 1e3, 1e4, 1e5, 1e6], (lambda v: {"N": v}), "photon pairs N"
        car, ss = [], []
        for i, v in enumerate(xs):
            o = model.sample(np.random.default_rng(seed_for(cfg.seed, "ch3study", which, v)), **kwf(v))
            car.append(o["extras"]["CAR"])
            ss.append(image_metrics(o["images"]["coincidence"], ref, model.sys["coincidence"], masks)["SSIM"])
            prog.progress((i + 1) / len(xs))
        prog.empty()
        a, b = st.columns(2)
        a.plotly_chart(line_chart(xs, {"CAR": car}, title=f"Coincidence-to-accidental ratio vs {xl}", xlab=xl,
                                  ylab="CAR", logy=True), width="stretch")
        b.plotly_chart(line_chart(xs, {"SSIM": ss}, title=f"SSIM vs {xl}", xlab=xl, ylab="SSIM"), width="stretch")

st.markdown("#### Matematiksel açıklamalar")
equation("1 · Biphoton state (SPDC, double-Gaussian approximation)",
         [r"|\Psi\rangle=\iint dx_s\,dx_i\;\Psi(x_s,x_i)\,|x_s\rangle|x_i\rangle",
          r"\Psi(x_s,x_i)=C\exp\!\left[-\frac{(x_s+x_i)^2}{4\sigma_S^2}-\frac{(x_s-x_i)^2}{4\sigma_D^2}\right]",
          r"K=\tfrac12\left(\frac{\sigma_S}{\sigma_D}+\frac{\sigma_D}{\sigma_S}\right)"],
         "- **Signal photon**: numuneden geçen foton. **Idler photon**: eşi; referans kolunda ilerler, numuneye dokunmaz.\n"
         "- σ_S pompa genişliğinden, σ_D faz eşleme (kristal uzunluğu) koşulundan gelir. 2-B durum x ve y çarpanlarının çarpımıdır.\n"
         "- Ψ, f(x_s)g(x_i) biçiminde çarpanlara ayrılmaz (Schmidt sayısı K ≫ 1) → uzaysal dolanıklık.\n"
         "- Yaklaşım: gerçek faz-eşleme sinc fonksiyonu bir Gauss ile değiştirilmiştir (Law & Eberly 2004).")
equation("Entanglement witness (EPR / Reid criterion)",
         [r"\Delta^2(x_s-x_i)\,\Delta^2(k_s+k_i)=\frac{\sigma_D^2}{\sigma_S^2}<\frac14\;\Rightarrow\;\text{EPR steering}\Rightarrow\text{entanglement}"],
         "Yakın alanda konum, uzak alanda momentum korelasyonu ölçülür. **Separable control** aynı yakın-alan konum korelasyonuna sahiptir "
         "fakat momentum korelasyonu yoktur (çarpım ≈ 1). Yüksek koinsidans oranı tek başına dolanıklık kanıtı değildir.")
equation("2–3 · Propagation and sample interaction (amplitude level)",
         [r"\psi(x_s|x_i)\propto\Psi(x_s,x_i)",
          r"v(x|x_i)=\mathcal{F}^{-1}\{H P\,\mathcal{F}[t\,\psi(\cdot|x_i)]\},\qquad P(X_s,x_i)=\sum_{x\in X_s}|v(x|x_i)|^2\le 1"],
         "Idler x_i'de algılandığında sinyal fotonu koşullu duruma indirgenir; t ile **genlik düzeyinde** çarpılır ve görüntüleme "
         "sisteminden geçer. Soğurma olasılığı azaltır (|t|²), faz gradyanları ışığı yönlendirir (pupil dışına → kayıp).")
equation("4–9 · Detection, losses, dark/background, true and accidental coincidences",
         [r"\Pr[\text{true coincidence at }X_s]=\eta_s\,\eta_i\,K(X_s)",
          r"\langle A\rangle\approx\frac{1}{M}\sum_X S_s(X)\,S_i(W(X))"],
         "Her üretilen çift algılanmaz: her foton η olasılıkla algılanır. Edinim M = N/μ koinsidans penceresine bölünür; arka plan ve "
         "karanlık sayımlar aynı toplam süreye yayılır. Tesadüfi koinsidanslar, aynı pencereye düşen ilişkisiz olaylardır ve "
         "gecikmeli pencere yöntemiyle tahmin edilir.")
equation("Coincidences vs G² vs g²",
         [r"C(X_s)=\#\{(s,i):\ \text{same window},\ X_i\in W(X_s)\}",
          r"G^{(2)}(x_s,x_i)=\langle I_s(x_s)I_i(x_i)\rangle\approx C/M",
          r"g^{(2)}(x_s,x_i)=\frac{\langle I_s(x_s)I_i(x_i)\rangle}{\langle I_s(x_s)\rangle\langle I_i(x_i)\rangle}=\frac{M\,C}{S_s S_i}"],
         "- **P(x_s,x_i)**: modelden gelen, üretilen çift başına ortak algılama olasılığı (teorik).\n"
         "- **C**: ölçülen ham koinsidans sayısı (gerçek + tesadüfi).\n"
         "- **G²**: pencere başına normalize edilmemiş ikinci derece korelasyon.\n"
         "- **g²**: tekil sayımlara göre normalize; ilişkisiz ışık için 1.")
equation("10 · Reconstruction", r"\hat T(X)=\frac{C(X)-A_{\rm delayed}(X)}{\eta_s\eta_i\,N_{\rm pairs}\,K_{\rm flat}(X)}",
         "K_flat: t = 1 için çift başına beklenen gerçek koinsidans olasılığı (flat-field kalibrasyonu).")
with st.expander("Varsayımlar ve sınırlamalar"):
    st.markdown("- Double-Gaussian biphoton; pompa FOV üzerinde düzgün kabul edilir (σ_S ≥ 400 µm için hata < %2).\n"
                "- Yalnızca yakın-alan (near-field) heralded-camera geometrisi; uzak-alan ghost geometrisi görüntülenmez.\n"
                "- Çoklu çift emisyonu bağımsız Poisson çiftleri olarak modellenir; zaman titreşimi pencere dışında yok.\n"
                "- Bu geometride koinsidans görüntüsü, aynı konum korelasyonlu **ayrılabilir** bir kaynakla da elde edilir "
                "(Doğrulama sayfası) → görüntü kalitesindeki herhangi bir fark dolanıklığa atfedilemez.\n"
                "- Arka plan bastırması, çiftlerin zaman korelasyonundan gelir; eşleşmiş bir kapılı (gated) darbeli klasik lazer simüle edilmemiştir.")
