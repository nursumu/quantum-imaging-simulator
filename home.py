import base64
from pathlib import Path

import streamlit as st

from ui.style import hero, panel, note

hero("Sentetik Biyolojik Hücrelerin Klasik ve Kuantum Görüntülenmesi için Etkileşimli Hesaplamalı Simülatör",
     "Koherent, Klasik Korelasyonlu ve Dolanık Foton Görüntüleme Yöntemlerinin Karşılaştırmalı İncelenmesi")

c1, c2 = st.columns([1.15, 1])
with c1:
    st.markdown("#### Araştırmanın amacı")
    st.markdown(
        "Aynı sentetik hücreyi, aynı optik sistemi, aynı dedektör verimliliğini, aynı arka plan akısını ve "
        "**aynı foton bütçesini** kullanarak üç görüntüleme yöntemini fiziksel olarak tutarlı modellerle "
        "simüle etmek ve görüntü kalitesini (SNR, CNR, SSIM, uzaysal çözünürlük) nicel olarak karşılaştırmak.")
    st.markdown("#### Araştırma sorusu")
    panel("<i>“Düşük foton sayısı ve gürültülü koşullar altında, korelasyon temelli kuantum görüntüleme "
          "yöntemleri sentetik biyolojik hücre görüntülerinin yeniden oluşturulmasında geleneksel koherent "
          "yoğunluk görüntülemeye kıyasla iyileşme sağlayabilir mi?”</i>")
    st.markdown("#### Bilimsel arka plan")
    st.markdown(
        "- **Shot-noise limit:** Koherent ışıkla yoğunluk ölçümünde foton sayımları Poisson dağılır; "
        "SNR ≈ √(algılanan foton). Bu, klasik ışık için standart kuantum sınırıdır.\n"
        "- **Ghost / correlation imaging:** İki ışın arasındaki yoğunluk korelasyonları, numuneyle etkileşmeyen "
        "ışının bilgisini de kullanarak görüntü oluşturabilir. Bu korelasyonlar *klasik* (termal ışık) veya "
        "*kuantum* (SPDC foton çiftleri) kökenli olabilir.\n"
        "- **Entanglement ≠ correlation:** Yüksek koinsidans oranı tek başına dolanıklığın kanıtı değildir. "
        "Dolanıklık, EPR (Reid) kriteri gibi bir *entanglement witness* ile gösterilmelidir.\n"
        "- **Adil karşılaştırma:** Kuantum yöntemin üstün olduğu varsayılmaz; avantaj olup olmadığı simülasyon "
        "sonuçlarından belirlenir.")
with c2:
    st.markdown("#### Görüntüleme sisteminin şeması")
    _svg = (Path(__file__).resolve().parents[1] / "ui" / "schematic.svg").read_bytes()
    st.markdown(f'<img src="data:image/svg+xml;base64,{base64.b64encode(_svg).decode()}" style="width:100%;border:1px solid #E1E5EC;border-radius:6px"/>',
                unsafe_allow_html=True)
    note("Şema ölçeksizdir. Üç kanal da aynı numune düzlemini, aynı NA'lı görüntüleme optiğini ve aynı "
         "piksel boyutlu dedektörleri kullanır.")

st.markdown("#### Üç görüntüleme kanalı")
cards = [
    ("Kanal 1 · Classical coherent intensity imaging", "Klasik Koherent Işık Şiddeti Görüntüleme",
     "Monokromatik koherent düzlem dalga → kompleks geçirgenlik t(x,y) → Angular Spectrum propagation → "
     "|U|² → Poisson foton sayımı. Referans (baseline) yöntem.", "app_pages/channel_1.py"),
    ("Kanal 2 · Classically correlated imaging", "Klasik Korelasyonlu Işıkla Görüntüleme",
     "Bölünmüş psödotermal (speckle) ışık: sinyal kolu hücreden geçer, referans kolu geçmez. Görüntü "
     "C_sr = ⟨ΔI_s ΔI_r⟩ çapraz kovaryansından oluşturulur. Dolanıklık içermez.", "app_pages/channel_2.py"),
    ("Kanal 3 · Entangled-photon quantum imaging", "Dolanık Fotonlarla Kuantum Görüntüleme",
     "SPDC biphoton durumu Ψ(x_s,x_i). Sinyal fotonu hücreyle etkileşir, idler referans kolunda algılanır. "
     "Görüntü gerçek − tesadüfi koinsidanslardan oluşturulur; EPR tanığı ve ayrılabilir kontrol içerir.",
     "app_pages/channel_3.py"),
]
cols = st.columns(3)
for col, (t, tr, d, page) in zip(cols, cards):
    with col:
        st.markdown(f'<div class="qi-card"><h4>{t}</h4><span class="qi-tag">{tr}</span><p style="margin-top:.5rem">{d}</p></div>',
                    unsafe_allow_html=True)
        st.page_link(page, label="Simülasyonu aç →", icon=":material/arrow_forward:")

st.markdown("#### Kullanım")
st.markdown(
    "1. Sol paneldeki **Genel simülasyon ayarları** tüm kanallar için ortaktır (dalga boyu, hücre, grid, foton bütçesi, "
    "dedektör verimliliği, arka plan, yayılım mesafesi, tekrar sayısı, tohum).\n"
    "2. **Sentetik Hücre** sayfasında ortak hücreyi inceleyin.\n"
    "3. Her kanal sayfasında kanala özgü parametreleri ayarlayıp simülasyonu çalıştırın.\n"
    "4. **Karşılaştırma** sayfasında üç kanalı aynı koşullarda çalıştırın ve taramalar yapın.\n"
    "5. **Doğrulama** sayfası, her kanalın bağımsız sayısal testlerini çalıştırır.")
st.info("Bu uygulama bir hesaplamalı modeldir. Gösterilen sonuçlar deneysel veri değildir ve deneysel "
        "olarak doğrulanmış biyolojik hücre görüntüleme sonuçları olarak yorumlanmamalıdır.", icon=":material/info:")
