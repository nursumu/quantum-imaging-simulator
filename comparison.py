from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from core.experiment import ESTIMATORS, MAIN_ESTIMATORS, evaluate, sweep, seed_for, ModelSet
from core.interpret import summary_sentences
from ui import cache
from ui.settings import physics_key
from ui.style import hero, show_heatmaps, line_chart, note, panel

RESULTS = Path(__file__).resolve().parents[1] / "results"
cfg = st.session_state.get("cfg") or __import__("ui.settings", fromlist=["x"]).global_settings()
hero("Karşılaştırmalı Analiz · Comparative analysis",
     "Aynı sentetik hücre, aynı optik, aynı η, aynı arka plan akısı ve aynı foton bütçesi tanımıyla üç kanal")

with st.container(border=True):
    st.markdown("**Kanal seçenekleri (karşılaştırma için)**")
    a, b, c, d = st.columns(4)
    M = a.select_slider("Ch2 frames M (reference acquisition)", [20, 50, 100, 200, 500], value=100)
    sp = b.slider("Ch2 speckle ℓ_c (µm)", 1.0, 6.0, 2.5, 0.5)
    sD = c.slider("Ch3 σ_D (µm)", 0.4, 3.0, 0.8, 0.1)
    mu = d.select_slider("Ch3 pairs per window μ", [0.01, 0.1, 1.0, 10.0], value=0.1)
ch2o = (("frames", M), ("speckle_um", sp))
ch3o = (("sigma_D", sD), ("mu", mu))
pk = physics_key(cfg)
panel(f"<b>Fairness:</b> budget mode <b>{cfg.budget_mode}</b> · N = {cfg.photons:.0e} · η = {cfg.qe} (all detectors) · "
      f"background {cfg.background} + dark {cfg.dark} counts/px over the <b>same acquisition time</b> (Ch2 spreads it over M frames, "
      f"Ch3 over its coincidence windows) · {cfg.repeats} repeats · seed {cfg.seed}")

tab1, tab2, tab3 = st.tabs(["Tek koşum: yan yana", "Taramalar (interaktif grafikler)", "Sonuç tablosu ve yorum"])

with tab1:
    if st.button("▶ Üç kanalı aynı koşullarda çalıştır", type="primary"):
        with st.status("Running the three channels ...", expanded=True) as status:
            st.write("Building / loading models (cell, coherent field, exact speckle moments, biphoton kernels) ...")
            ms = cache.bind_set(cache.model_set(pk, ch2o, ch3o), cfg)
            st.write("Sampling photon statistics and computing metrics ...")
            rows = evaluate(ms, cfg.photons, cfg.noise_counts, repeats=cfg.repeats, tag="single", resolution=True)
            imgs = {}
            for ch, est in [("ch1", "coherent"), ("ch2", "cov"), ("ch3", "coincidence")]:
                o = ms.models[ch].sample(np.random.default_rng(seed_for(cfg.seed, "gallery", ch)))
                imgs[ch] = o["images"][est]
            status.update(label="Done", state="complete")
        st.session_state["cmp_single"] = (rows, imgs, ms.ref, cfg)
    if "cmp_single" in st.session_state:
        rows, imgs, ref, c0 = st.session_state["cmp_single"]
        show_heatmaps([dict(img=ref, title="Original synthetic cell (T_ref)", zmin=0, zmax=1.2),
                       dict(img=imgs["ch1"], title="Ch1 · coherent intensity", zmin=0, zmax=1.2),
                       dict(img=imgs["ch2"], title="Ch2 · covariance C_sr image", zmin=0, zmax=1.2),
                       dict(img=imgs["ch3"], title="Ch3 · coincidence image", zmin=0, zmax=1.2)], c0.dx_det, cols=4, height=300)
        tab = []
        for r in rows:
            tab.append({"Method": r["label"], "SNR (noise)": f"{r['SNR_noise']:.3g} ± {r['SNR_noise_std']:.2g}",
                        "SNR (vs object)": f"{r['SNR_object']:.3g}", "CNR": f"{r['CNR']:.3g} ± {r['CNR_std']:.2g}",
                        "SSIM": f"{r['SSIM']:.3f} ± {r['SSIM_std']:.3f}",
                        "Resolution 10–90 % (µm)": "not resolvable" if not np.isfinite(r.get("resolution_um", np.nan)) else f"{r['resolution_um']:.2f}",
                        "Photons on sample": f"{r['x_N_incident']:.3g}",
                        "Total detected photons": f"{r['x_detected_total']:.4g}",
                        "Frames / windows": f"{r['x_frames']:.3g}",
                        "Background (counts/px)": f"{r['noise']:.3g}",
                        "RMSE vs object": f"{r['RMSE_object']:.3f}"})
        st.dataframe(pd.DataFrame(tab), hide_index=True, width="stretch")
        note("SNR (noise) uses the channel's own noise-free expectation; SNR (vs object) and SSIM compare with the true cell. "
             "Ch3 'frames' are coincidence windows (M = N/μ).")

with tab2:
    st.markdown("Taramalar mevcut genel ayarlarla çalıştırılır (her nokta için global tekrar sayısı). Hazır sonuçlar referans "
                "konfigürasyonla `python scripts/run_sweeps.py` komutu ile üretilmiştir.")
    src = st.radio("Data source", ["Run now with current settings", "Load precomputed reference results (results/)"],
                   horizontal=True)
    kinds = {"photons": ("Photon number N", [1e2, 1e3, 1e4, 1e5, 1e6], True),
             "noise": ("Background b (counts/px/acquisition)", [0.1, 1.0, 10.0, 100.0, 1000.0], True),
             "thickness": ("Cell thickness h (µm)", [2.0, 5.0, 10.0], False),
             "time": ("Measurement time T (relative; N and b ∝ T)", [0.01, 0.1, 1.0, 10.0], True)}
    if src.startswith("Run"):
        sel = st.multiselect("Sweeps", list(kinds), default=["photons", "noise"],
                             format_func=lambda k: kinds[k][0])
        reps = st.slider("Repeats per point", 1, 5, min(cfg.repeats, 2))
        if st.button("▶ Taramaları çalıştır", type="primary"):
            prog = st.progress(0.0, "starting ...")
            ms = cache.bind_set(cache.model_set(pk, ch2o, ch3o), cfg)
            allrows = []
            for i, k in enumerate(sel):
                def cb(f, msg, i=i):
                    prog.progress((i + f) / len(sel), f"{kinds[k][0]}: {msg}")
                allrows += sweep(cfg, k, kinds[k][1], dict(ch2o), dict(ch3o), repeats=reps, progress=cb,
                                 base_ms=None if k == "thickness" else ms)
            prog.empty()
            st.session_state["cmp_sweeps"] = pd.DataFrame(allrows)
    else:
        f = RESULTS / "sweeps_reference.csv"
        if f.exists():
            st.session_state["cmp_sweeps"] = pd.read_csv(f)
            meta = RESULTS / "sweeps_reference_meta.json"
            if meta.exists():
                import json
                mj = json.loads(meta.read_text())
                cf = mj["config"]
                st.caption(f"Reference configuration: λ₀ = {cf['wavelength_um']} µm, NA = {cf['NA']}, grid {cf['grid']}², "
                           f"FOV {cf['fov_um']} µm, h = {cf['thickness_um']} µm, η = {cf['qe']}, b = {cf['background']}, "
                           f"budget '{cf['budget_mode']}', N = {cf['photons']:.0e} (default), Ch2 {mj['ch2']}, Ch3 {mj['ch3']}, "
                           f"{mj['repeats']} repeats, runtime {mj['runtime_s']} s.")
        else:
            st.warning("results/sweeps_reference.csv not found — run `python scripts/run_sweeps.py`.")
    df = st.session_state.get("cmp_sweeps")
    if df is not None and len(df):
        show_all = st.toggle("Show all Ch2 estimators (signal-arm and reference-normalised)", value=False)
        ests = ESTIMATORS if show_all else MAIN_ESTIMATORS
        for k in [k for k in kinds if k in set(df["tag"])]:
            d = df[df["tag"] == k]
            st.markdown(f"##### {kinds[k][0]}")
            metrics = ["SNR_noise", "CNR", "SSIM"] + (["resolution_um"] if k == "photons" and "resolution_um" in d else [])
            cols = st.columns(len(metrics))
            for col, met in zip(cols, metrics):
                ser, err, xs = {}, {}, None
                for lab, ch, est in ests:
                    g = d[(d["channel"] == ch) & (d["estimator"] == est)].sort_values("x_value")
                    if len(g):
                        xs = g["x_value"].tolist()
                        ser[lab] = g[met].tolist()
                        err[lab] = g.get(met + "_std", pd.Series([0] * len(g))).fillna(0).tolist()
                ylab = {"SNR_noise": "SNR", "resolution_um": "10–90 % edge width (µm)"}.get(met, met)
                col.plotly_chart(line_chart(xs, ser, err, title=f"{ylab} vs {kinds[k][0].split(' (')[0]}",
                                            xlab=kinds[k][0], ylab=ylab, logx=kinds[k][2],
                                            logy=met == "SNR_noise", height=380,
                                            colors={e[0]: i for i, e in enumerate(ESTIMATORS)}), width="stretch")
            if k == "photons":
                note("Okuma notu: SNR < 1 veya SSIM < 0.02 olan noktalarda görüntüler esasen gürültüdür; sıralama anlamlı değildir. "
                     "Çok düşük N'de kovaryans tahmini ≈ 0 veya tanımsızdır (referans varyansı D ≤ 0); bu durumda SNR = ortalama/RMS "
                     "tanımı ≈ 1'e yakın yapay bir değer verebilir. Çözünürlük, kenar ölçülemediğinde gösterilmez.")

with tab3:
    df = st.session_state.get("cmp_sweeps")
    if df is None or not len(df):
        st.info("Önce ‘Taramalar’ sekmesinde taramaları çalıştırın veya hazır sonuçları yükleyin.")
    else:
        st.markdown("#### Sonuç tablosu")
        show = df[[(c, e) in {(x[1], x[2]) for x in ESTIMATORS} for c, e in zip(df["channel"], df["estimator"])]]
        cols = ["tag", "x_value", "label", "SNR_noise", "SNR_object", "CNR", "SSIM", "resolution_um",
                "x_detected_total", "x_frames", "noise", "RMSE_object"]
        cols = [c for c in cols if c in show]
        st.dataframe(show[cols].rename(columns={"tag": "sweep", "x_value": "value", "x_detected_total": "detected photons",
                                                "x_frames": "frames/windows", "noise": "background"}).round(4),
                     hide_index=True, width="stretch", height=380)
        st.download_button("CSV indir", df.to_csv(index=False).encode(), "comparison_results.csv", "text/csv")
        st.markdown("#### Otomatik, veri temelli yorum (SSIM; fark > 2σ ise 'daha iyi')")
        for k in sorted(set(df["tag"])):
            with st.expander(f"Sweep: {k}", expanded=(k == "photons")):
                for line in summary_sentences(df[df["tag"] == k], k, "SSIM"):
                    st.markdown("- " + line)
        st.markdown("#### Bilimsel yorum")
        st.markdown(
            "- **Shot-noise baseline.** Aynı numune dozunda ve düşük arka planda koherent yoğunluk görüntüleme (Kanal 1), "
            "her algılanan fotonu kullandığı için genellikle en yüksek SNR'ı verir. Bu, klasik ışık için beklenen standart sınırdır.\n"
            "- **Koinsidans görüntüleme (Kanal 3)** her yararlı olay için idler fotonunun da algılanmasını gerektirir "
            "(η_i < 1 ve koinsidans penceresi) → düşük arka planda SNR kaybı. Arka plan sinyale yaklaştığında veya aştığında, zaman "
            "korelasyonu ilişkisiz sayımları bastırır ve kanal avantaj kazanabilir. Bu avantaj, aynı konum korelasyonuna sahip "
            "ayrılabilir bir kaynakla da elde edildiğinden **uzaysal dolanıklığa atfedilemez**; Cauchy–Schwarz ihlali klasik olmayan "
            "zaman (foton sayısı) korelasyonunu gösterir.\n"
            "- **Klasik kovaryans (Kanal 2)**, kare başına piksel başına foton sayısı ≪ 1 olduğunda shot-noise baskın hale gelir. "
            "Arka plandan bağımsız olması bir özelliktir fakat varyansı yüksektir.\n"
            "- **Çözünürlük** tüm kanallarda NA ve piksel boyutuyla sınırlıdır; bu modelde hiçbir kanal çözünürlük avantajı sağlamaz.\n"
            "- **Sınırlamalar:** ince numune yaklaşımı, ideal dedektörler, double-Gaussian biphoton, near-field geometri; eşleşmiş "
            "kapılı (gated) darbeli klasik lazer simüle edilmemiştir. Sonuçlar deneysel veri değildir.")
