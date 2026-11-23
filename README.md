FP-DM-[KELAS]-[NO]-README
=========================
Final Project Data Mining - Klasifikasi Ketidakpuasan Pelanggan dari Ulasan Produk Tokopedia

ISI BERKAS
----------
FP-DM-[KELAS]-[NO]-REPORT.pdf          Laporan (Bab I-V)
FP-DM-[KELAS]-[NO]-DATASET.csv         Dataset MENTAH (sheet tokopedia_products_with_review), sebelum Data Preparation
FP-DM-[KELAS]-[NO]-KODEPROGRAM-1.ipynb Notebook lengkap: Data Understanding -> Data Preparation -> Modeling -> Evaluation,
                                       sekaligus mengekspor folder artifacts/ (tabel, kurva, model) untuk Streamlit
FP-DM-[KELAS]-[NO]-KODEPROGRAM-2.py    Aplikasi Streamlit (salinan app.py)
FP-DM-[KELAS]-[NO]-LINKSTREAMLIT.pdf   Tautan aplikasi Streamlit
streamlit_repo/                        Folder siap di-push ke GitHub untuk Streamlit Community Cloud
                                       (app.py, requirements.txt, artifacts/)

CARA MENJALANKAN NOTEBOOK (KODEPROGRAM-1)
-----------------------------------------
1. Python 3.10+; pip install pandas numpy scikit-learn scipy matplotlib seaborn jupyter
2. Letakkan notebook dan berkas DATASET.csv dalam SATU folder, lalu ubah variabel DATA_PATH
   di sel "Setup" menjadi nama berkas CSV Anda (default: FP-DM-X-00-DATASET.csv).
3. Jalankan semua sel (Run All). Waktu eksekusi +/- 6-8 menit.
4. Hasil: folder artifacts/ (CSV, model.joblib, summary.json, figures/).

CARA MENJALANKAN STREAMLIT
--------------------------
LOKAL : cd streamlit_repo && pip install -r requirements.txt && streamlit run app.py
ONLINE: push folder streamlit_repo ke repo GitHub publik -> share.streamlit.io -> New app ->
        pilih repo, branch, main file path = app.py -> Deploy -> salin URL ke LINKSTREAMLIT.pdf

CATATAN
-------
- Dataset pada Excel memiliki batas 32.767 karakter per sel; sebagian list ulasan terpotong. Notebook memulihkan
  elemen yang masih utuh (lihat laporan Bab III dan IV).
- Semua angka di laporan dan Streamlit berasal dari artifacts/summary.json hasil notebook yang sama.
