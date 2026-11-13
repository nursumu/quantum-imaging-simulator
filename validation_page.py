import time

import pandas as pd
import streamlit as st

from core.config import SimConfig
from ui.style import hero, note
from validation import test_channel_1, test_channel_2, test_channel_3

hero("Doğrulama · Validation", "Her kanal için bağımsız sayısal testler — sınır durumları, foton bütçesi, Poisson istatistiği, "
     "yakınsama, kontrol deneyleri")
st.markdown(
    "Bu testler **matematiksel modelin ve sayısal uygulamanın** doğruluğunu kontrol eder (analitik çözümler, korunum yasaları, "
    "istatistiksel tutarlılık). Deneysel doğrulama **değildir**. Aynı testler komut satırından `pytest validation/` ile de çalışır.")
use_cur = st.toggle("Use current global settings (default: reference configuration)", value=False,
                    help="Kapalıyken testler varsayılan referans konfigürasyonla (λ=810 nm, NA=0.4, 128², 64 µm) çalışır.")
cfg = st.session_state.get("cfg") or __import__("ui.settings", fromlist=["x"]).global_settings() if use_cur else SimConfig()

suites = [("Kanal 1 · Coherent", test_channel_1, "~1 s"), ("Kanal 2 · Classical correlation", test_channel_2, "~40 s"),
          ("Kanal 3 · Entangled photons", test_channel_3, "~60 s")]
cols = st.columns(3)
for col, (name, mod, dur) in zip(cols, suites):
    with col:
        if st.button(f"▶ {name} ({dur})", width="stretch"):
            prog = st.progress(0.0, "running ...")
            t0 = time.time()
            res = mod.run_checks(cfg, progress=lambda f: prog.progress(min(f, 1.0)))
            prog.empty()
            st.session_state[f"val_{name}"] = (res, time.time() - t0)
for name, mod, _ in suites:
    if f"val_{name}" not in st.session_state:
        continue
    res, dt = st.session_state[f"val_{name}"]
    n_ok = sum(r["passed"] for r in res)
    st.markdown(f"#### {name} — {n_ok}/{len(res)} passed ({dt:.0f} s)")
    (st.success if n_ok == len(res) else st.error)(
        "All tests passed." if n_ok == len(res) else "Some tests failed — inspect the table.")
    df = pd.DataFrame([{"Result": "✅ PASS" if r["passed"] else "❌ FAIL", "Category": r["category"], "Test": r["test"],
                        "Measured": r["value"], "Criterion": r["criterion"], "Meaning": r["details"]} for r in res])
    st.dataframe(df, hide_index=True, width="stretch")
note("Kategoriler: boundary (sınır durumları) · physics (analitik fizik) · budget (foton bütçesi) · statistics (Poisson ve "
     "tahminci istatistiği) · convergence (sayısal yakınsama) · control (kontrol deneyleri) · entanglement (dolanıklık tanığı) · noise.")
