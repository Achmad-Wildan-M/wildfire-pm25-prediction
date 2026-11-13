# Prediksi PM2.5 akibat Kebakaran Hutan

Proyek Machine Learning untuk memprediksi konsentrasi PM2.5 pada wilayah terdampak kebakaran hutan menggunakan data pengamatan kebakaran berbasis satelit dan faktor meteorologi.

## Gambaran Umum

Kebakaran hutan merupakan salah satu faktor lingkungan yang dapat menyebabkan peningkatan pencemaran udara, terutama melalui pelepasan partikel halus (PM2.5). Konsentrasi PM2.5 yang tinggi dapat berdampak negatif terhadap kesehatan manusia dan menurunkan kualitas udara.

Proyek ini bertujuan untuk mengembangkan model Machine Learning guna memprediksi konsentrasi PM2.5 pada wilayah terdampak kebakaran hutan dengan mengintegrasikan data pengamatan kebakaran berbasis satelit dan data meteorologi.

Proyek ini dikembangkan sebagai bagian dari tugas mata kuliah Data Science untuk mengeksplorasi pemanfaatan Machine Learning dalam mendukung prediksi kualitas udara dan pemantauan lingkungan.

## Tujuan Proyek

Tujuan utama dari proyek ini adalah:

- Memprediksi konsentrasi PM2.5 pada wilayah terdampak kebakaran hutan menggunakan Machine Learning.
- Mengintegrasikan data pengamatan kebakaran hutan berbasis satelit dengan data meteorologi.
- Menganalisis hubungan antara aktivitas kebakaran hutan, kondisi meteorologi, dan konsentrasi PM2.5.
- Membandingkan kinerja beberapa model Machine Learning.
- Mengidentifikasi model yang menghasilkan prediksi konsentrasi PM2.5 paling akurat.

## Sustainable Development Goals (SDGs)

Proyek ini mendukung:

**SDG 3 – Kehidupan Sehat dan Sejahtera (Good Health and Well-Being)**

**Target 3.9:** Mengurangi secara signifikan jumlah kematian dan penyakit akibat bahan kimia berbahaya serta pencemaran udara, air, dan tanah.

Melalui prediksi konsentrasi PM2.5, proyek ini diharapkan dapat berkontribusi dalam meningkatkan pemahaman dan pemantauan pencemaran udara pada wilayah terdampak kebakaran hutan.

## Sumber Dataset

Proyek ini menggunakan data dari beberapa sumber berikut:

### 1. NASA FIRMS

[Fire Information for Resource Management System (FIRMS)](https://firms.modaps.eosdis.nasa.gov/)

NASA FIRMS menyediakan data pengamatan titik panas (hotspot) atau kebakaran aktif berbasis satelit, yang mencakup:

- Lokasi kebakaran (latitude dan longitude).
- Tanggal dan waktu deteksi.
- Fire Radiative Power (FRP).
- Suhu kecerahan (brightness temperature).
- Tingkat kepercayaan deteksi kebakaran (fire detection confidence).

Data tersebut akan digunakan untuk merepresentasikan aktivitas kebakaran hutan.

### 2. Copernicus Atmosphere Monitoring Service (CAMS)

[CAMS Global Reanalysis (EAC4)](https://ads.atmosphere.copernicus.eu/datasets/cams-global-reanalysis-eac4)

CAMS menyediakan data komposisi atmosfer dan meteorologi, yang mencakup:

- Konsentrasi PM2.5.
- Suhu dan variabel meteorologi lainnya yang tersedia.
- Kondisi atmosfer yang berkaitan dengan kualitas udara.

Data konsentrasi PM2.5 dari CAMS akan digunakan sebagai target prediksi dalam proyek ini.

**Catatan:** CAMS menyediakan data berbasis reanalisis. Oleh karena itu, nilai PM2.5 yang digunakan dalam proyek ini merupakan estimasi konsentrasi PM2.5 dari CAMS dan tidak selalu merepresentasikan pengukuran langsung dari stasiun pemantauan kualitas udara.

## Metodologi

Tahapan pengerjaan proyek ini terdiri dari beberapa proses berikut:

1. **Pengumpulan Data (Data Collection)**

   Mengumpulkan data pengamatan kebakaran hutan dari NASA FIRMS serta data atmosfer dan meteorologi dari CAMS.

2. **Pra-pemrosesan Data (Data Preprocessing)**

   Membersihkan, memfilter, dan mempersiapkan dataset agar dapat digunakan dalam proses analisis dan pemodelan.

3. **Integrasi Data (Data Integration)**

   Menggabungkan data aktivitas kebakaran hutan dengan data meteorologi dan PM2.5 berdasarkan informasi lokasi dan waktu.

4. **Analisis Data Eksploratif (Exploratory Data Analysis / EDA)**

   Mengeksplorasi distribusi data, pola, korelasi, serta hubungan antara aktivitas kebakaran hutan dan konsentrasi PM2.5.

5. **Rekayasa Fitur (Feature Engineering)**

   Menyiapkan fitur-fitur yang relevan untuk proses Machine Learning, termasuk variabel aktivitas kebakaran hutan dan meteorologi.

6. **Pengembangan Model (Model Development)**

   Melatih dan mengembangkan beberapa model Machine Learning untuk melakukan prediksi konsentrasi PM2.5.

7. **Evaluasi Model (Model Evaluation)**

   Membandingkan kinerja setiap model menggunakan metrik evaluasi regresi yang sesuai.

## Model Machine Learning

Proyek ini akan membandingkan tiga pendekatan Machine Learning berikut:

| Model | Kategori | Tujuan |
|---|---|---|
| Linear Regression | Baseline | Menjadi acuan awal kinerja prediksi |
| Random Forest Regressor | Machine Learning | Menangkap hubungan nonlinier dalam data |
| Multilayer Perceptron (MLP) | Deep Learning | Mempelajari pola yang lebih kompleks dari fitur masukan |

Model MLP akan menggunakan minimal dua hidden layer.

## Metrik Evaluasi

Kinerja setiap model akan dievaluasi menggunakan beberapa metrik regresi berikut:

- **Mean Absolute Error (MAE):** Mengukur rata-rata selisih absolut antara nilai aktual dan nilai hasil prediksi.
- **Root Mean Squared Error (RMSE):** Mengukur kesalahan prediksi dengan memberikan penalti lebih besar terhadap kesalahan yang bernilai tinggi.
- **R-squared (R²):** Mengukur seberapa baik model dapat menjelaskan variasi pada variabel target.

## Struktur Repository

```text
wildfire-pm25-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│
├── results/
│   ├── figures/
│   └── metrics/
│
└── docs/
    └── references.md
```

## Ruang Lingkup Proyek

Proyek ini berfokus pada prediksi konsentrasi PM2.5 di wilayah terdampak kebakaran hutan dengan memanfaatkan data pengamatan kebakaran berbasis satelit dan variabel meteorologi.

Cakupan wilayah geografis, periode pengamatan, serta konfigurasi akhir dataset akan ditentukan pada tahap pengumpulan dan pra-pemrosesan data.

Proyek ini berfokus pada prediksi dan tidak mengasumsikan bahwa seluruh konsentrasi PM2.5 yang teramati secara eksklusif disebabkan oleh kebakaran hutan. Sumber pencemaran lainnya juga dapat berkontribusi terhadap tingkat PM2.5.

## Anggota Kelompok

Proyek ini dikembangkan oleh Kelompok 12 sebagai bagian dari tugas mata kuliah Data Science.

| No. | Nama | NIM |
|---|---|---|
| 1 | Achmad Wildan Miftakhudin | 244311032 |
| 2 | Alfiansyah Wahyu Pratama | 244311034 |
| 3 | Irham Najib Azimul Qowi | 244311045 |

## Status Proyek

**Direncanakan (Planned)**

Proyek saat ini berada pada tahap awal pengembangan dengan fokus pada eksplorasi dataset, pengumpulan data, pra-pemrosesan, dan pengembangan model.

## Referensi

1. Mirzaei, M., Bertazzon, S., Couloigner, I., & Farjad, B. (2021). Assessing the Potential of Artificial Intelligence (Artificial Neural Networks) in Predicting the Spatiotemporal Pattern of Wildfire-Generated PM2.5 Concentration. *Geomatics, 1*(1), 18–33. https://doi.org/10.3390/geomatics1010003

2. Vu, B. N., Bi, J., Wang, W., Huff, A., Kondragunta, S., & Liu, Y. (2022). Application of geostationary satellite and high-resolution meteorology data in estimating hourly PM2.5 levels during the Camp Fire episode in California. *Remote Sensing of Environment, 271*, 112890. https://doi.org/10.1016/j.rse.2022.112890

3. Raffuse, S., O'Neill, S., & Schmidt, R. (2024). A model for rapid PM2.5 exposure estimates in wildfire conditions using routinely available data: rapidfire v0.1.3. *Geoscientific Model Development, 17*, 381–397. https://doi.org/10.5194/gmd-17-381-2024

---

**Disclaimer:** Proyek ini dikembangkan untuk tujuan pendidikan dan penelitian. Hasil prediksi yang dihasilkan tidak dapat dianggap sebagai pengukuran resmi kualitas udara maupun sebagai dasar diagnosis atau saran medis.
