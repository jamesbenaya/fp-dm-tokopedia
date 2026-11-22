"""FP Data Mining - Streamlit: Klasifikasi Ketidakpuasan Pelanggan dari Ulasan Produk Tokopedia.
Semua angka/grafik dibaca dari folder artifacts/ yang dihasilkan oleh notebook KODEPROGRAM-1 (konsisten dengan laporan)."""
import json, re
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ART = Path(__file__).parent / "artifacts"
st.set_page_config(page_title="FP Data Mining - Ulasan Tokopedia", page_icon="📊", layout="wide")

# ---------- fungsi cleaning: HARUS identik dengan notebook ----------
SLANG = {"tdk": "tidak", "gak": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak", "gk": "tidak",
         "gx": "tidak", "yg": "yang", "bgt": "banget", "bngt": "banget", "sy": "saya", "udh": "sudah",
         "udah": "sudah", "sdh": "sudah", "blm": "belum", "krn": "karena", "karna": "karena",
         "dgn": "dengan", "utk": "untuk", "jg": "juga", "tp": "tapi", "sm": "sama", "brg": "barang",
         "mantul": "mantap", "trims": "terima kasih", "thx": "terima kasih", "thanks": "terima kasih"}
_url = re.compile(r"https?://\S+|www\.\S+")
_nonalnum = re.compile(r"[^a-z0-9\s]")
_repeat = re.compile(r"([a-z])\1{2,}")


def clean_text(t):
    t = str(t).lower()
    t = _url.sub(" ", t)
    t = _nonalnum.sub(" ", t)
    t = _repeat.sub(r"\1", t)
    return " ".join(SLANG.get(w, w) for w in t.split())


@st.cache_data
def load_json():
    return json.load(open(ART / "summary.json", encoding="utf-8"))


@st.cache_data
def load_csv(name, **kw):
    return pd.read_csv(ART / name, **kw)


@st.cache_resource
def load_model():
    return joblib.load(ART / "model.joblib")


S = load_json()
RED, BLUE = "#C44E52", "#4C72B0"

st.sidebar.title("📊 FP Data Mining")
st.sidebar.caption("Klasifikasi ketidakpuasan pelanggan dari teks ulasan produk Tokopedia")
page = st.sidebar.radio("Halaman", ["1. Ringkasan", "2. EDA", "3. Data Preparation", "4. Modeling & Evaluasi",
                                    "5. Interpretasi & Error Analysis", "6. Coba Prediksi"])
st.sidebar.markdown("---")
st.sidebar.caption("Sumber data: Tokopedia Products with Review (lihat laporan). Seluruh angka dibaca dari artefak notebook.")

# =============================================================== 1
if page.startswith("1"):
    st.title("Prediksi Ketidakpuasan Pelanggan dari Ulasan Tokopedia")
    st.markdown("**Task:** Klasifikasi biner teks — `Tidak puas` (rating 1–3) vs `Puas` (rating 4–5).  \n"
                "**Framework:** CRISP-DM · **Pembagian data:** per produk (anti-leakage) · **Metrik utama:** F1 & PR-AUC kelas tidak puas.")
    m = S["test_metrics_thr"]
    c = st.columns(5)
    c[0].metric("Ulasan bersih", f"{S['n_reviews']:,}")
    c[1].metric("Produk", f"{S['n_products']:,}")
    c[2].metric("% Tidak puas", f"{S['pct_tidak_puas']:.1%}")
    c[3].metric("F1 (tidak puas, test)", f"{m['f1']:.3f}")
    c[4].metric("PR-AUC (test)", f"{m['pr_auc']:.3f}")
    c = st.columns(3)
    c[0].metric("Recall (test)", f"{m['recall']:.3f}")
    c[1].metric("Precision (test)", f"{m['precision']:.3f}")
    c[2].metric("Akurasi (test)", f"{m['accuracy']:.3f}", help="Hanya pendamping: pada data imbalanced akurasi menyesatkan.")
    st.info(f"Model final: **{S['final_model']}** + TF-IDF (unigram+bigram). "
            f"Baseline 'selalu menebak Puas' menghasilkan akurasi {S['dummy']['accuracy']:.1%} tetapi F1 kelas tidak puas = 0 — "
            "itulah alasan akurasi bukan ukuran utama.")

# =============================================================== 2
elif page.startswith("2"):
    st.title("Exploratory Data Analysis")
    tab = st.tabs(["Target & Rating", "Panjang Teks", "Kategori", "Penjual & Harga", "Tren Waktu", "Kata Khas"])
    with tab[0]:
        rd = pd.DataFrame({"rating": list(S["rating_dist"].keys()), "jumlah": list(S["rating_dist"].values())})
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(rd, x="rating", y="jumlah", title="Distribusi rating", color_discrete_sequence=[BLUE]), width="stretch")
        cc = pd.DataFrame({"kelas": ["Puas (4-5)", "Tidak puas (1-3)"],
                           "jumlah": [S["class_counts"]["puas"], S["class_counts"]["tidak_puas"]]})
        c2.plotly_chart(px.bar(cc, x="kelas", y="jumlah", title="Distribusi kelas target (imbalanced)", color="kelas",
                               color_discrete_sequence=[BLUE, RED]), width="stretch")
        st.caption("Kelas tidak puas adalah minoritas → dipakai F1/PR-AUC, `class_weight=balanced`, dan threshold terpilih.")
    with tab[1]:
        smp = load_csv("reviews_sample.csv.gz")
        smp["kelas"] = smp.tidak_puas.map({0: "Puas", 1: "Tidak puas"})
        st.plotly_chart(px.histogram(smp[smp.n_kata <= 60], x="n_kata", color="kelas", barmode="overlay", histnorm="probability density",
                                     nbins=30, title="Panjang ulasan (kata) — sampel seimbang 12 ribu ulasan",
                                     color_discrete_map={"Puas": BLUE, "Tidak puas": RED}), width="stretch")
        st.write("Rata-rata & median jumlah kata per kelas (data penuh):")
        st.dataframe(pd.DataFrame(S["len_stats"]).rename(index=str).rename(columns={"0": "Puas", "1": "Tidak puas"}))
    with tab[2]:
        cat = load_csv("eda_category.csv")
        cat["pct"] = cat.pct_tidak_puas * 100
        st.plotly_chart(px.bar(cat.sort_values("pct"), x="pct", y="main_category", orientation="h", color_discrete_sequence=[RED],
                               title="% ulasan tidak puas per kategori (n ≥ 2.000)", labels={"pct": "% tidak puas", "main_category": ""},
                               height=550), width="stretch")
    with tab[3]:
        c1, c2 = st.columns(2)
        sel = load_csv("eda_seller.csv", index_col=0)
        sel["pct"] = sel.pct_tidak_puas * 100
        c1.plotly_chart(px.bar(sel.reset_index(), x="index", y="pct", title="% tidak puas menurut tipe penjual",
                               labels={"index": "", "pct": "% tidak puas"}, color_discrete_sequence=[RED]), width="stretch")
        pr = load_csv("eda_price.csv")
        pr["Kuintil harga"] = [f"Q{i+1}" for i in range(len(pr))]
        pr["pct"] = pr.pct_tidak_puas * 100
        c2.plotly_chart(px.bar(pr, x="Kuintil harga", y="pct", title="% tidak puas menurut kuintil harga (Q1 termurah)",
                               labels={"pct": "% tidak puas"}, color_discrete_sequence=["#55A868"]), width="stretch")
        st.dataframe(pr[["bin_harga", "n", "pct_tidak_puas"]])
    with tab[4]:
        mo = load_csv("eda_monthly.csv", parse_dates=["bulan"])
        mo["pct"] = mo.pct_tidak_puas * 100
        st.plotly_chart(px.line(mo, x="bulan", y="pct", markers=True, title="Tren bulanan % ulasan tidak puas",
                                labels={"pct": "% tidak puas", "bulan": ""}), width="stretch")
    with tab[5]:
        n, p = load_csv("eda_terms_neg.csv"), load_csv("eda_terms_pos.csv")
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(n.sort_values("log_odds"), x="log_odds", y="term", orientation="h", title="Kata khas ulasan TIDAK puas",
                               color_discrete_sequence=[RED], height=550), width="stretch")
        p["skor"] = -p["log_odds"]
        c2.plotly_chart(px.bar(p.sort_values("skor"), x="skor", y="term", orientation="h", title="Kata khas ulasan PUAS",
                               color_discrete_sequence=[BLUE], height=550), width="stretch")

# =============================================================== 3
elif page.startswith("3"):
    st.title("Data Preparation")
    st.markdown("Satu baris data mentah = satu **produk** dengan list ulasan berbentuk string → diubah menjadi satu baris per **ulasan**. "
                f"Sebanyak **{S['products_truncated']:,} produk** memiliki list yang terpotong di batas sel Excel (32.767 karakter); "
                "elemen yang masih utuh dipulihkan.")
    log = pd.DataFrame(S["prep_log"], columns=["Langkah", "Jumlah ulasan"])
    log["Jumlah ulasan"] = log["Jumlah ulasan"].astype(int)
    st.dataframe(log, width="stretch", hide_index=True)
    fig = go.Figure(go.Funnel(y=log["Langkah"], x=log["Jumlah ulasan"], marker={"color": BLUE}))
    st.plotly_chart(fig, width="stretch")
    st.markdown("""
**Keputusan utama & alasan**
- Duplikat `review_id` & duplikat teks dibuang → mencegah ulasan identik muncul di train dan test (*leakage*).
- Stopword **tidak** dibuang: kata negasi (*tidak, belum*) adalah sinyal ketidakpuasan.
- Fitur `review_response`, `review_like`, `bad_rating_reason`, `rating_average` **tidak dipakai** (bocoran target).
- Pembagian train/test 80/20 **per produk** (`GroupShuffleSplit`), TF-IDF di-fit hanya pada data latih.
""")
    st.write("Pembagian data:")
    st.dataframe(pd.DataFrame(S["split"], columns=["Data", "Ulasan", "Produk", "% tidak puas"]), hide_index=True)

# =============================================================== 4
elif page.startswith("4"):
    st.title("Modeling & Evaluasi")
    st.subheader("Perbandingan model — validasi silang GroupKFold(3) pada data latih")
    st.dataframe(load_csv("model_comparison.csv").round(4), width="stretch", hide_index=True)
    st.subheader("Perbandingan model — test set (belum pernah dilihat)")
    st.dataframe(load_csv("model_comparison_test.csv").round(4), width="stretch", hide_index=True)
    st.caption(f"Model teks final untuk deployment: **{S['final_model']}**. Selisih PR-AUC model +metadata vs teks-saja (OOF): {S['meta_gain_prauc']:+.4f}.")

    st.subheader("Hasil akhir pada test set")
    cm = S["confusion_matrix"]
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.imshow(cm, text_auto=",", x=["Puas", "Tidak puas"], y=["Puas", "Tidak puas"], color_continuous_scale="Blues",
                              labels=dict(x="Prediksi", y="Aktual"), title="Confusion matrix"), width="stretch")
    pr, roc = load_csv("pr_curve.csv"), load_csv("roc_curve.csv")
    f = go.Figure(go.Scatter(x=pr.recall, y=pr.precision, mode="lines", line=dict(color=RED)))
    f.update_layout(title=f"Precision–Recall (PR-AUC={S['test_metrics_thr']['pr_auc']:.3f})", xaxis_title="Recall", yaxis_title="Precision")
    f.add_hline(y=S["test_base_rate"], line_dash="dash", annotation_text="baseline acak")
    c2.plotly_chart(f, width="stretch")

    st.subheader("Eksplorasi threshold")
    th = load_csv("threshold_curve.csv")
    thr_sel = st.slider("Threshold skor", float(th.threshold.min()), float(th.threshold.max()), float(S["threshold"]), step=0.005)
    row = th.iloc[(th.threshold - thr_sel).abs().argmin()]
    c = st.columns(3)
    c[0].metric("Precision", f"{row.precision:.3f}"); c[1].metric("Recall", f"{row.recall:.3f}"); c[2].metric("F1", f"{row.f1:.3f}")
    ft = go.Figure()
    for col, color in [("precision", BLUE), ("recall", RED), ("f1", "#55A868")]:
        ft.add_trace(go.Scatter(x=th.threshold, y=th[col], name=col, line=dict(color=color)))
    ft.add_vline(x=S["threshold"], line_dash="dash", annotation_text="threshold terpilih (OOF)")
    ft.update_layout(title="Trade-off precision–recall terhadap threshold", xaxis_title="threshold")
    st.plotly_chart(ft, width="stretch")

    st.subheader("Kinerja per kategori produk (test)")
    st.dataframe(load_csv("eval_per_category.csv").round(3), width="stretch", hide_index=True)

# =============================================================== 5
elif page.startswith("5"):
    st.title("Interpretasi & Error Analysis")
    c1, c2 = st.columns(2)
    fn, fp = load_csv("feat_neg.csv"), load_csv("feat_pos.csv")
    c1.plotly_chart(px.bar(fn.sort_values("koef"), x="koef", y="fitur", orientation="h", title="Fitur pendorong TIDAK PUAS",
                           color_discrete_sequence=[RED], height=600), width="stretch")
    fp["koef"] = -fp["koef"]
    c2.plotly_chart(px.bar(fp.sort_values("koef"), x="koef", y="fitur", orientation="h", title="Fitur pendorong PUAS",
                           color_discrete_sequence=[BLUE], height=600), width="stretch")
    st.subheader("Tema keluhan pada ulasan tidak puas (leksikon sederhana)")
    th = load_csv("themes.csv")
    th["pct"] = th.pct_ulasan_tidak_puas * 100
    st.plotly_chart(px.bar(th.sort_values("pct"), x="pct", y="tema", orientation="h", color_discrete_sequence=[RED],
                           labels={"pct": "% ulasan tidak puas yang menyebut tema", "tema": ""}), width="stretch")
    st.dataframe(th.round(3), hide_index=True, width="stretch")
    st.subheader("Contoh kesalahan model (test set)")
    err = load_csv("error_examples.csv")
    jenis = st.radio("Jenis kesalahan", sorted(err.jenis.unique()), horizontal=True)
    st.dataframe(err[err.jenis == jenis].head(30), width="stretch", hide_index=True)
    st.write("Persentase ulasan yang diprediksi *tidak puas* menurut rating asli (test):")
    st.dataframe(load_csv("pred_by_rating.csv").round(4), hide_index=True)

# =============================================================== 6
else:
    st.title("Coba Prediksi Ulasan")
    st.caption("Masukkan ulasan dalam Bahasa Indonesia. Model memakai threshold hasil validasi silang.")
    ex = {"(pilih contoh)": "", "Positif": "Barang sesuai deskripsi, packing rapi, pengiriman cepat. Terima kasih!",
          "Keluhan kualitas": "Barang rusak pas sampai, tidak sesuai foto, kecewa banget",
          "Salah kirim": "Pesan warna hitam yang datang putih, salah kirim. Penjual tidak merespon"}
    pick = st.selectbox("Contoh", list(ex.keys()))
    text = st.text_area("Teks ulasan", value=ex[pick], height=120)
    if st.button("Prediksi", type="primary") and text.strip():
        model = load_model()
        proba = model.predict_proba([clean_text(text)])[0, 1] if hasattr(model, "predict_proba") else None
        score = proba if proba is not None else model.decision_function([clean_text(text)])[0]
        label = "TIDAK PUAS" if score >= S["threshold"] else "PUAS"
        (st.error if label == "TIDAK PUAS" else st.success)(f"Prediksi: **{label}** (skor {score:.3f}; threshold {S['threshold']:.3f})")
        st.write("Teks setelah cleaning:", f"`{clean_text(text)}`")
        tf, clf = model.named_steps["tfidf"], model.named_steps["clf"]
        if hasattr(clf, "coef_"):
            X = tf.transform([clean_text(text)])
            names = tf.get_feature_names_out()
            contrib = pd.DataFrame({"fitur": names[X.indices], "kontribusi": X.data * clf.coef_.ravel()[X.indices]})
            st.write("Fitur yang paling memengaruhi (positif = ke arah *tidak puas*):")
            st.dataframe(contrib.reindex(contrib.kontribusi.abs().sort_values(ascending=False).index).head(10), hide_index=True)
