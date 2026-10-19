# Analisis Skalabilitas Dataset untuk Titanic dengan Benchmark ML

## Latar Belakang

Proyek ini bertujuan untuk mengeksplorasi bagaimana ukuran dataset memengaruhi waktu preprocessing dan pelatihan model Machine Learning, dengan kata kunci utama yaitu skalabilitas komputasi. Inspirasi utamanya adalah pendekatan High Performance Computing (HPC), di mana performa sistem dapat berubah ketika beban kerja bertambah secara nyata. Namun, benchmark pada laptop atau komputer pribadi tidak setara dengan pengujian pada cluster atau superkomputer. Dataset yang digunakan adalah Titanic dari file `data/train.csv` dan `data/test.csv` yang sudah tersedia di repositori. Tidak ada pengunduhan dataset baru, dan tidak ada perubahan pada dataset asli.

## Tujuan

- Memeriksa karakteristik dataset Titanic secara eksploratif.
- Menangani missing value dan fitur kategorikal dengan pendekatan yang aman.
- Menggunakan stratified train-validation split untuk evaluasi yang konsisten.
- Menganalisis pengaruh ukuran data terhadap waktu pemrosesan dan pelatihan.
- Membandingkan model Logistic Regression dan Random Forest.
- Menyimpan hasil benchmark ke folder `output/` dalam bentuk CSV, grafik PNG, dan ringkasan TXT.

## Struktur Folder

```text
hpc-flood-risk-analysis/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
├── output/
│   ├── benchmark_results.csv
│   ├── runtime_comparison.png
│   ├── metrics_comparison.png
│   └── benchmark_summary.txt
├── benchmark.py
├── requirements.txt
├── README.md
├── .gitignore
└── .venv/
```

## Instalasi

1. Pastikan Python 3.10+ sudah terinstal.
2. Buat environment virtual (opsional tetapi disarankan).
3. Instal dependensi:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Cara Menjalankan

```bash
python benchmark.py
```

Program akan:

- membaca dataset Titanic dari `data/train.csv`;
- melakukan profiling data awal;
- membagi data menjadi train dan validation dengan stratifikasi;
- menjalankan benchmark pada ukuran data 25%, 50%, 75%, dan 100%;
- membandingkan Logistic Regression dan Random Forest;
- menyimpan hasil CSV, grafik PNG, dan ringkasan insight ke folder `output/`.

## Metodologi

### 1. Eksplorasi Data

Program akan memeriksa:

- jumlah baris dan kolom;
- tipe data setiap kolom;
- missing values per fitur;
- duplikasi baris;
- distribusi target `Survived`.

### 2. Preprocessing Aman

Untuk mencegah data leakage, semua transformasi dilakukan hanya pada subset training. Pipeline preprocessing mencakup:

- imputasi nilai numerik dengan median;
- imputasi fitur kategorikal dengan nilai paling sering;
- encoding fitur kategorikal dengan OneHotEncoder.

### 3. Train-Validation Split

Validasi memperhatikan kelas target menggunakan `train_test_split(..., stratify=y, random_state=42)`.

### 4. Benchmark Ukuran Data

Ukuran data dilatih dalam skenario berikut:

- 25% dari data training
- 50% dari data training
- 75% dari data training
- 100% dari data training

Untuk setiap kombinasi ukuran data dan model, program mengukur:

- waktu preprocessing dengan `time.perf_counter()`;
- waktu pelatihan model dengan `time.perf_counter()`;
- total waktu eksekusi.

### 5. Evaluasi

Model dievaluasi pada validation set yang tidak digunakan saat training. Metrik yang dicatat:

- accuracy
- precision
- recall
- F1-score

## Hasil yang Dihasilkan

Setelah program dijalankan, folder `output/` berisi:

- `benchmark_results.csv` — tabel hasil seluruh eksperimen
- `runtime_comparison.png` — grafik perbandingan waktu eksekusi antar model
- `metrics_comparison.png` — grafik perbandingan metrik evaluasi
- `benchmark_summary.txt` — ringkasan insight dan kesimpulan otomatis

## Keterbatasan

Benchmark ini dilakukan pada dataset kecil dan lingkungan komputer pribadi. Ini bukan pengujian superkomputer, bukan demonstrasi dari infrastruktur HPC produksi, dan tidak membuktikan percepatan HPC secara langsung. Hasil yang ditampilkan lebih berfungsi sebagai demonstrasi konseptual tentang skalabilitas komputasi, bukan sebagai bukti performa sistem berskala besar.

## Kesimpulan

Proyek ini memperlihatkan bahwa ukuran dataset secara umum memengaruhi waktu preprocessing dan training. Dalam konteks pembelajaran mesin, perubahan ukuran data dapat memengaruhi biaya komputasi secara proporsional dan membantu memahami kebutuhan paralelisasi, distribusi komputasi, dan optimasi pipeline pada lingkungan HPC. Namun, interpretasi hasil harus disesuaikan dengan keterbatasan perangkat dan dataset yang digunakan.

## Catatan Penting

Semua angka hasil eksperimen diproses dari eksekusi program. Tidak ada angka yang dibuat secara manual.
