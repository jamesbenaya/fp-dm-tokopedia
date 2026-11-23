# Final Project Data Mining – Klasifikasi Ketidakpuasan Pelanggan Tokopedia

Proyek Data Mining untuk mengklasifikasikan ketidakpuasan pelanggan berdasarkan ulasan produk Tokopedia. Proyek ini mencakup tahapan pengolahan data hingga pembangunan model klasifikasi dan implementasi aplikasi berbasis web menggunakan **Streamlit**.

Tahapan pengerjaan meliputi:

- **Data Understanding** untuk memahami karakteristik dan struktur dataset ulasan produk Tokopedia.
- **Data Preparation** untuk membersihkan dan mempersiapkan data sebelum proses pemodelan.
- **Modeling** untuk membangun model klasifikasi ketidakpuasan pelanggan.
- **Evaluation** untuk mengukur dan mengevaluasi performa model.
- **Deployment** untuk mengimplementasikan hasil klasifikasi ke dalam aplikasi Streamlit.

Final Project mata kuliah Data Mining, Kelompok E-08, 2026.

## Dataset

Dataset yang digunakan merupakan data produk dan ulasan pelanggan Tokopedia dalam format CSV.

File dataset utama:

`FP-DM-E-08-DATASET.csv`

Dataset tersebut merupakan dataset mentah sebelum melalui proses Data Preparation. Data digunakan sebagai dasar untuk melakukan eksplorasi, pembersihan, pengolahan, dan pemodelan klasifikasi.

**Catatan:** Dataset yang berasal dari Excel memiliki batas maksimal 32.767 karakter per sel sehingga sebagian daftar ulasan mengalami pemotongan. Notebook dilengkapi mekanisme pemulihan elemen ulasan yang masih utuh. Pembahasan lebih lanjut terdapat pada Bab III dan Bab IV laporan.

## Tahapan Pengerjaan

Proyek ini terdiri dari empat tahapan utama Data Mining:

1. **Data Understanding:** Melakukan eksplorasi dataset untuk mengetahui struktur data, jenis atribut, karakteristik data, serta kondisi awal dataset.
2. **Data Preparation:** Melakukan proses pembersihan dan persiapan data agar dapat digunakan dalam proses pemodelan.
3. **Modeling:** Melakukan pembangunan model Machine Learning untuk mengklasifikasikan ketidakpuasan pelanggan berdasarkan data ulasan.
4. **Evaluation:** Mengevaluasi hasil pemodelan untuk mengetahui performa dan kemampuan model dalam melakukan klasifikasi.

Seluruh tahapan tersebut dikerjakan menggunakan notebook Jupyter yang tersedia pada file `FP-DM-E-08-KODEPROGRAM-1.ipynb`.

## Fitur Aplikasi

Aplikasi Streamlit digunakan untuk menampilkan hasil pengolahan dan pemodelan Data Mining dalam bentuk antarmuka web.

Fitur aplikasi meliputi:

- Menampilkan hasil pengolahan dan analisis dataset.
- Menampilkan informasi hasil pemodelan klasifikasi.
- Menampilkan hasil evaluasi performa model.
- Menyajikan visualisasi hasil analisis.
- Menggunakan model yang telah disimpan untuk mendukung proses klasifikasi.

## Hasil

Hasil akhir proyek mencakup:

- Dataset hasil Data Preparation.
- Model klasifikasi yang telah dibangun.
- Hasil evaluasi performa model.
- Grafik dan visualisasi hasil analisis.
- Aplikasi Streamlit untuk menampilkan hasil klasifikasi.

Seluruh angka yang ditampilkan pada laporan dan aplikasi Streamlit berasal dari file `artifacts/summary.json` yang dihasilkan melalui notebook yang sama, sehingga hasil keduanya menggunakan sumber perhitungan yang konsisten.

## Menjalankan Notebook

Pastikan Python versi 3.10 atau lebih baru sudah terinstal.

Instal library yang dibutuhkan:

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn jupyter
```

Langkah menjalankan notebook:

1. Letakkan file notebook dan dataset CSV dalam satu folder.
2. Buka `FP-DM-E-08-KODEPROGRAM-1.ipynb`.
3. Pastikan variabel `DATA_PATH` pada bagian Setup sesuai dengan nama file dataset.

```python
DATA_PATH = "FP-DM-E-08-DATASET.csv"
```

4. Jalankan seluruh sel menggunakan fitur **Run All**.
5. Tunggu hingga proses pengolahan dan pemodelan selesai.

Estimasi waktu eksekusi sekitar 6–8 menit, bergantung pada spesifikasi komputer.

Setelah proses selesai, notebook akan menghasilkan folder `artifacts/` yang berisi file CSV hasil pengolahan, model klasifikasi (`model.joblib`), ringkasan hasil analisis (`summary.json`), serta grafik dan visualisasi pada folder `figures/`.

## Menjalankan Aplikasi Streamlit

Masuk ke direktori aplikasi:

```bash
cd streamlit_repo
```

Instal seluruh dependency:

```bash
pip install -r requirements.txt
```

Jalankan aplikasi:

```bash
streamlit run app.py
```

Aplikasi akan terbuka melalui browser pada alamat lokal yang ditampilkan oleh terminal, umumnya:

`http://localhost:8501`

Pastikan folder `artifacts/` sudah tersedia sebelum menjalankan aplikasi.

## Deploy Online (Streamlit Community Cloud)

Untuk menjalankan aplikasi secara online, ikuti langkah berikut:

1. Upload folder `streamlit_repo` ke repository GitHub publik.
2. Buka [Streamlit Community Cloud](https://share.streamlit.io).
3. Login menggunakan akun GitHub.
4. Pilih **New App**.
5. Pilih repository dan branch yang digunakan.
6. Isi Main File Path dengan `app.py`.
7. Klik **Deploy**.
8. Setelah aplikasi berhasil dijalankan, salin URL aplikasi dan simpan pada dokumen `FP-DM-E-08-LINKSTREAMLIT.pdf`.

## Struktur File

```text
.
├── FP-DM-E-08-REPORT.pdf
├── FP-DM-E-08-DATASET.csv
├── FP-DM-E-08-KODEPROGRAM-1.ipynb
├── FP-DM-E-08-KODEPROGRAM-2.py
├── FP-DM-E-08-LINKSTREAMLIT.pdf
│
└── streamlit_repo/
    ├── app.py
    ├── requirements.txt
    └── artifacts/
        ├── model.joblib
        ├── summary.json
        ├── file pendukung lainnya
        └── figures/
```

Keterangan:

- `REPORT.pdf` berisi laporan akhir proyek Bab I–V.
- `DATASET.csv` merupakan dataset mentah sebelum Data Preparation.
- `KODEPROGRAM-1.ipynb` merupakan notebook lengkap proses Data Mining.
- `KODEPROGRAM-2.py` merupakan salinan source code aplikasi Streamlit.
- `LINKSTREAMLIT.pdf` berisi tautan aplikasi yang telah di-deploy.
- `streamlit_repo/` merupakan folder aplikasi yang siap diunggah ke GitHub.

## Anggota Kelompok

| Nama | NIM |
|---|---|
| Adriano Femaz Rivaldy | 23082010139 |
| James Petra Benaya Prawira Nugroho | 23082010081 |
