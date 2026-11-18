# LAPORAN PROYEK MACHINE LEARNING
## Wildfire PM2.5 Prediction

---

## INFORMASI PROYEK

**Judul Proyek:** Prediksi Konsentrasi PM2.5 pada Wilayah Terdampak Kebakaran Hutan Menggunakan Machine Learning  
**Nama Kelompok:** Kelompok 12  
**Anggota 1:** Achmad Wildan Miftakhudin  
**NIM:** 244311032  
**Anggota 2:** Alfiansyah Wahyu Pratama  
**NIM:** 244311034  
**Anggota 3:** Irham Najib Azimul Qowi  
**NIM:** 244311045  
**Program Studi:** Teknologi Rekayasa Perangkat Lunak  
**Mata Kuliah:** Data Science  
**Dosen Pengampu:** Gus Nanang Syaifuddiin, S.Kom., M.Kom.,
**Tahun Akademik:** Semester 5 (Ganjil 2026/2027)  
**Link GitHub Repository:** https://github.com/Achmad-Wildan-M/wildfire-pm25-prediction
**Link Video Pembahasan:** [Masukkan URL video jika sudah tersedia]  
**SDG yang Dipilih:** SDG 3 – Kehidupan Sehat dan Sejahtera (Good Health and Well-Being)  
**Target SDG yang Relevan:** Target 3.9 – Mengurangi secara signifikan jumlah kematian dan penyakit akibat bahan kimia berbahaya serta pencemaran udara, air, dan tanah.

---

## 0. KERANGKA ML LIFECYCLE (L-M-O-P)

Laporan ini mengikuti empat tahap ML Lifecycle, yaitu Learning Problem (L), Model Design (M), Optimization (O), dan Predict & Evaluate (P).

| Tahap | Fokus pada proyek |
|---|---|
| **Learning Problem [L]** | Merumuskan prediksi konsentrasi PM2.5 sebagai permasalahan regresi serta mengumpulkan dan menyiapkan data kebakaran hutan dan meteorologi. |
| **Model Design [M]** | Merancang Decision Tree Regressor, XGBoost Regressor, dan Gated Recurrent Unit (GRU). |
| **Optimization [O]** | Melatih model, menentukan loss function, melakukan tuning hyperparameter, dan mengendalikan overfitting. |
| **Predict & Evaluate [P]** | Mengevaluasi prediksi menggunakan MAE, RMSE, dan R² serta membandingkan ketiga model. |

Alur proyek:

`Learning Problem [L] → Model Design [M] → Optimization [O] → Predict & Evaluate [P]`

Hasil evaluasi akan digunakan untuk menentukan perbaikan pada tahap sebelumnya, misalnya memperbaiki fitur, preprocessing, arsitektur, atau hyperparameter.

---

## 1. LEARNING OUTCOMES

Melalui proyek ini, mahasiswa diharapkan mampu:

1. **[L]** Mengidentifikasi permasalahan pencemaran udara akibat kebakaran hutan dan menghubungkannya dengan SDG 3 Target 3.9.
2. **[L]** Merumuskan prediksi konsentrasi PM2.5 sebagai permasalahan regresi.
3. **[L]** Mengumpulkan, mengeksplorasi, dan mengintegrasikan data aktivitas kebakaran hutan serta data atmosfer.
4. **[L]** Melakukan preprocessing dan feature engineering pada data yang memiliki dimensi waktu dan lokasi.
5. **[M]** Merancang tiga model, yaitu Decision Tree Regressor, XGBoost Regressor, dan GRU.
6. **[O]** Melatih serta mengoptimasi model menggunakan konfigurasi dan hyperparameter yang sesuai.
7. **[P]** Mengevaluasi dan membandingkan performa model menggunakan MAE, RMSE, dan R².
8. Mendokumentasikan proses eksperimen dan mengunggah kode proyek ke GitHub.

---

## 2. PROJECT OVERVIEW [L]

### 2.1 Latar Belakang

Kebakaran hutan merupakan salah satu faktor lingkungan yang dapat menyebabkan penurunan kualitas udara. Proses pembakaran vegetasi menghasilkan asap dan berbagai polutan, termasuk partikulat halus atau Particulate Matter 2.5 (PM2.5). Partikel PM2.5 memiliki ukuran sangat kecil sehingga dapat masuk ke saluran pernapasan dan menimbulkan dampak buruk bagi kesehatan manusia.

Konsentrasi PM2.5 di wilayah terdampak kebakaran dapat berubah berdasarkan aktivitas kebakaran dan kondisi atmosfer. Aktivitas kebakaran dapat direpresentasikan melalui informasi titik panas (hotspot) dan Fire Radiative Power (FRP). Sementara itu, kondisi meteorologi seperti suhu, kelembapan, dan angin dapat memengaruhi penyebaran serta konsentrasi polutan di atmosfer.

Perkembangan teknologi penginderaan jauh dan layanan data atmosfer memungkinkan pengumpulan informasi kebakaran serta kualitas udara dalam cakupan wilayah dan periode yang luas. Data tersebut dapat dimanfaatkan sebagai masukan untuk membangun model Machine Learning yang mempelajari hubungan antara aktivitas kebakaran, kondisi meteorologi, dan konsentrasi PM2.5.

Berdasarkan permasalahan tersebut, proyek ini bertujuan mengembangkan model Machine Learning untuk memprediksi konsentrasi PM2.5 pada wilayah terdampak kebakaran hutan. Data aktivitas kebakaran direncanakan diperoleh dari NASA Fire Information for Resource Management System (FIRMS), sedangkan data PM2.5 dan variabel atmosfer diperoleh dari Copernicus Atmosphere Monitoring Service (CAMS) Global Reanalysis EAC4.

Proyek ini juga akan membandingkan tiga pendekatan, yaitu Decision Tree Regressor sebagai model baseline, XGBoost Regressor sebagai model Machine Learning lanjutan, dan Gated Recurrent Unit (GRU) sebagai model Deep Learning yang memanfaatkan urutan waktu. Perbandingan dilakukan untuk mengetahui kemampuan masing-masing model dalam memprediksi konsentrasi PM2.5.

### 2.2 Keterkaitan dengan SDGs (WAJIB)

| Komponen | Jawaban |
|---|---|
| **SDG yang dipilih** | SDG 3 – Kehidupan Sehat dan Sejahtera (Good Health and Well-Being). |
| **Target SDG** | Target 3.9, yaitu mengurangi kematian dan penyakit akibat bahan kimia berbahaya serta pencemaran udara, air, dan tanah. |
| **Masalah nyata** | Kebakaran hutan dapat menghasilkan asap dan partikulat PM2.5 yang menurunkan kualitas udara serta berpotensi membahayakan kesehatan masyarakat di wilayah terdampak. |
| **Peran ML** | Machine Learning digunakan untuk mempelajari pola hubungan antara aktivitas kebakaran, kondisi meteorologi, dan konsentrasi PM2.5 sehingga dapat menghasilkan estimasi konsentrasi PM2.5 berdasarkan fitur yang tersedia. |
| **Indikator kontribusi** | Performa prediksi konsentrasi PM2.5 yang diukur menggunakan MAE, RMSE, dan R². Hasilnya diharapkan dapat mendukung pemahaman dan pemantauan kualitas udara, bukan menggantikan sistem pemantauan resmi. |
| **Pertimbangan etika** | Data yang digunakan berasal dari sumber terbuka dan tidak memuat identitas pribadi. Keterbatasan cakupan spasial, temporal, dan estimasi reanalysis perlu diperhatikan. Hasil prediksi tidak boleh dianggap sebagai pengukuran resmi atau dasar diagnosis medis. |

**Sumber ide:** Isu pencemaran udara akibat kebakaran hutan dan potensi dampaknya terhadap kesehatan masyarakat, serta peluang pemanfaatan data satelit dan Machine Learning untuk pemantauan lingkungan.

### 2.3 Ringkasan Alur Lifecycle Proyek

Proyek dimulai dengan merumuskan target prediksi berupa konsentrasi PM2.5 dan mengumpulkan data hotspot dari NASA FIRMS serta data atmosfer dari CAMS. Selanjutnya, data akan dibersihkan, diselaraskan berdasarkan lokasi dan waktu, serta dieksplorasi untuk memahami karakteristiknya. Tiga model kemudian dirancang dan dilatih menggunakan data yang telah dipersiapkan. Performa model dievaluasi menggunakan MAE, RMSE, dan R². Hasil evaluasi akan menjadi dasar perbaikan pada tahap pengumpulan data, rekayasa fitur, desain model, maupun optimasi.

---

## 3. BUSINESS UNDERSTANDING / PROBLEM UNDERSTANDING [L]

### 3.1 Problem Statements

1. Aktivitas kebakaran hutan dapat menghasilkan partikulat PM2.5 yang berdampak terhadap kualitas udara dan kesehatan manusia.
2. Konsentrasi PM2.5 dipengaruhi oleh berbagai faktor, termasuk aktivitas kebakaran dan kondisi meteorologi, sehingga hubungan antarvariabel perlu dianalisis.
3. Data aktivitas kebakaran dan data atmosfer berasal dari sumber yang berbeda sehingga perlu diselaraskan berdasarkan lokasi dan waktu sebelum digunakan untuk pemodelan.
4. Diperlukan perbandingan beberapa pendekatan Machine Learning untuk mengetahui model yang paling baik dalam memprediksi konsentrasi PM2.5 pada data yang digunakan.

### 3.2 Formulasi Learning Problem (WAJIB)

| Komponen | Jawaban |
|---|---|
| **Target** | Konsentrasi PM2.5 dalam satuan µg/m³. Jenis tugas adalah regresi karena model memprediksi nilai numerik kontinu. |
| **Objective** | Meminimalkan kesalahan prediksi antara nilai PM2.5 aktual menurut dataset target dan nilai hasil prediksi model. Loss utama yang direncanakan adalah Mean Squared Error (MSE), sedangkan evaluasi menggunakan MAE, RMSE, dan R². |
| **Data** | Data hotspot dan atribut kebakaran dari NASA FIRMS serta PM2.5 dan variabel atmosfer dari CAMS EAC4. Data berbentuk tabular dengan atribut spasial dan temporal. Periode dan wilayah akhir ditetapkan setelah pemeriksaan ketersediaan serta kelengkapan data. |

**Asumsi data:** Data dari lokasi dan waktu yang berdekatan dapat memiliki keterkaitan. Oleh sebab itu, data tidak langsung diasumsikan i.i.d. Pembagian training, validation, dan test direncanakan berdasarkan urutan waktu untuk mengurangi kebocoran informasi temporal.

### 3.3 Goals

1. Mengumpulkan dan mengintegrasikan data aktivitas kebakaran hutan dari NASA FIRMS dengan data PM2.5 dan atmosfer dari CAMS.
2. Menyiapkan dataset tabular yang memiliki kesesuaian lokasi dan waktu serta target PM2.5 yang jelas.
3. Mengembangkan tiga model regresi, yaitu Decision Tree Regressor, XGBoost Regressor, dan GRU.
4. Membandingkan performa ketiga model menggunakan MAE, RMSE, dan R² pada data pengujian.
5. Mengeksplorasi apakah penambahan fitur aktivitas kebakaran, seperti jumlah hotspot dan FRP, dapat meningkatkan performa prediksi dibandingkan penggunaan fitur meteorologi saja.
6. Menghasilkan dokumentasi dan kode eksperimen yang dapat dijalankan kembali.

### 3.4 Solution Approach [M]

#### Model 1 – Baseline Model

**Nama Model:** Decision Tree Regressor.

**Teori Singkat:** Decision Tree Regressor membagi ruang fitur secara berulang berdasarkan nilai ambang (threshold) pada fitur tertentu hingga terbentuk sejumlah daerah (leaf). Prediksi pada setiap leaf adalah rata-rata nilai target pada daerah tersebut, dan pemisahan dipilih untuk mengurangi kesalahan kuadrat.

**Alasan Pemilihan:** Decision Tree dipilih sebagai baseline karena sederhana, tidak memerlukan scaling fitur, mampu menangkap hubungan nonlinier dasar, dan mudah diinterpretasikan melalui aturan percabangan. Kedalaman pohon akan dibatasi agar baseline tidak overfit. Model ini juga menjadi pembanding langsung bagi model ensemble berbasis pohon.

#### Model 2 – Advanced / ML Model

**Nama Model:** XGBoost Regressor (Extreme Gradient Boosting).

**Teori Singkat:** XGBoost merupakan metode ensemble berbasis gradient boosting. Pohon keputusan dibangun secara berurutan, dan setiap pohon baru dilatih untuk memperbaiki kesalahan (residual) dari gabungan pohon sebelumnya. Prediksi akhir diperoleh dari penjumlahan kontribusi seluruh pohon, dengan learning rate dan regularisasi untuk mengendalikan kompleksitas.

**Alasan Pemilihan:** Model ini dapat mempelajari hubungan nonlinier dan interaksi antarfitur, serta sesuai untuk data tabular yang memiliki beragam variabel lingkungan dan fitur riwayat (lag).

**Keunggulan:**
- Dapat menangkap pola nonlinier dan interaksi antarfitur.
- Memiliki regularisasi bawaan serta parameter pengendali kompleksitas, seperti learning rate, kedalaman pohon, dan subsampling.
- Menyediakan informasi feature importance sebagai salah satu alat interpretasi.

**Kelemahan:**
- Memiliki banyak hyperparameter sehingga proses tuning membutuhkan waktu.
- Dapat mengalami overfitting jika jumlah dan kedalaman pohon tidak dikendalikan.
- Tidak memodelkan urutan waktu secara eksplisit; informasi historis diberikan melalui fitur lag.

#### Model 3 – Deep Learning Model (WAJIB)

**Nama Model:** Gated Recurrent Unit (GRU).

**Jenis Deep Learning:** Recurrent Neural Network (RNN) varian GRU untuk data deret waktu multivariat.

**Teori Singkat:** GRU merupakan varian RNN yang menggunakan mekanisme gerbang (update gate dan reset gate) untuk mengatur informasi yang dipertahankan atau dilupakan pada hidden state di setiap langkah waktu. Dengan mekanisme tersebut, GRU dapat mempelajari ketergantungan temporal pada rangkaian masukan dengan jumlah parameter yang lebih sedikit dibandingkan LSTM.

**Rencana Arsitektur Awal:**
- Input layer: rangkaian (sequence) berbentuk (window, jumlah fitur); rencana awal menggunakan window 8 langkah waktu (24 jam) dengan interval tiga jam.
- GRU layer: 64 unit.
- Dense layer: 32 neuron, aktivasi ReLU.
- Output layer: 1 neuron untuk menghasilkan prediksi PM2.5 pada langkah waktu terakhir.
- Regularisasi: dropout atau L2 akan dipertimbangkan untuk mengendalikan overfitting.

**Alasan Pemilihan:** Aktivitas kebakaran dan kondisi angin pada beberapa jam sebelumnya dapat memengaruhi konsentrasi PM2.5 saat ini. GRU dirancang untuk mempelajari pola temporal tersebut dan memenuhi kebutuhan penggunaan model Deep Learning.

---

## 4. DATA UNDERSTANDING [L]

### 4.1 Informasi Dataset

Proyek ini direncanakan menggunakan dua sumber dataset utama.

#### Dataset 1 – NASA FIRMS

**Sumber:** NASA Fire Information for Resource Management System (FIRMS)  
**URL:** https://firms.modaps.eosdis.nasa.gov/download/  
**Jenis data:** Data tabular/geospasial dengan informasi waktu dan lokasi deteksi kebakaran.  
**Format yang tersedia:** CSV dan format geospasial lain sesuai produk yang dipilih.

Informasi yang akan digunakan meliputi:
- Latitude dan longitude: koordinat lokasi deteksi.
- Tanggal dan waktu deteksi.
- Brightness temperature: suhu kecerahan pada deteksi kebakaran.
- Fire Radiative Power (FRP): indikator energi radiasi kebakaran.
- Confidence: tingkat keyakinan deteksi, jika tersedia pada produk yang digunakan.

#### Dataset 2 – Copernicus CAMS Global Reanalysis EAC4

**Sumber:** Copernicus Atmosphere Monitoring Service (CAMS)  
**URL:** https://ads.atmosphere.copernicus.eu/datasets/cams-global-reanalysis-eac4  
**Jenis data:** Data atmosfer dan meteorologi berbasis grid spasial serta deret waktu.  
**Format:** Format data ilmiah seperti NetCDF atau format lain yang disediakan layanan.

Variabel yang direncanakan:
- PM2.5: target prediksi.
- Suhu: variabel kondisi atmosfer.
- Komponen angin U dan V: merepresentasikan arah/komponen pergerakan angin.
- Kelembapan relatif atau variabel kelembapan yang tersedia dan sesuai.

**Catatan ketersediaan:** CAMS EAC4 merupakan data reanalysis dengan cakupan historis global. Berdasarkan katalog ADS, cakupan temporal EAC4 adalah 2003 hingga 2025 dengan resolusi waktu tiga jam, sehingga rencana periode 2018–2023 berada dalam cakupan tersebut. Ketersediaan setiap variabel, resolusi, satuan, dan periode harus diverifikasi pada katalog serta dokumentasi dataset sebelum pengunduhan.

#### Rencana cakupan data

- Wilayah studi: [Ditentukan setelah pemeriksaan ketersediaan dan kelengkapan data; California merupakan kandidat awal].
- Periode: Rencana awal 2018–2023, menyesuaikan ketersediaan dan kesesuaian produk data.
- Resolusi waktu: Mengikuti resolusi CAMS yang digunakan; EAC4 menyediakan data dengan interval waktu tiga jam.
- Resolusi spasial: Mengikuti grid CAMS EAC4, yaitu sekitar 0,75° × 0,75° (± 80 km).
- Jumlah baris, jumlah kolom, dan ukuran file: [Diisi setelah data diunduh dan diaudit].

### 4.2 Deskripsi Fitur

Tabel berikut merupakan rancangan fitur yang akan dibentuk setelah proses integrasi. Nama kolom final dapat disesuaikan dengan hasil preprocessing.

| Nama Fitur | Tipe Data | Deskripsi | Sumber |
|---|---|---|---|
| timestamp | Datetime | Waktu pengamatan atau agregasi | FIRMS/CAMS |
| hour | Integer | Jam (UTC) dari timestamp, mewakili pola harian | Turunan timestamp |
| month | Integer | Bulan dari timestamp, mewakili pola musiman | Turunan timestamp |
| latitude | Float | Koordinat lintang grid pengamatan | FIRMS/CAMS |
| longitude | Float | Koordinat bujur grid pengamatan | FIRMS/CAMS |
| hotspot_count | Integer | Jumlah deteksi hotspot pada grid dan periode tertentu | FIRMS |
| avg_frp | Float | Rata-rata Fire Radiative Power pada grid dan periode tertentu | FIRMS |
| temperature | Float | Suhu atmosfer | CAMS |
| relative_humidity | Float | Kelembapan relatif, jika tersedia atau dapat dihitung secara valid | CAMS |
| wind_u | Float | Komponen angin arah timur-barat | CAMS |
| wind_v | Float | Komponen angin arah utara-selatan | CAMS |
| pm25 | Float | Konsentrasi PM2.5 sebagai target prediksi | CAMS |

**Catatan:** Daftar fitur final bergantung pada produk, variabel, dan satuan yang benar-benar berhasil diunduh. Konversi satuan PM2.5 harus mengikuti metadata variabel CAMS. Jika nilai tersedia dalam kg/m³, konversi ke µg/m³ dilakukan dengan mengalikan nilai dengan 10⁹. Fitur riwayat (lag) dan jendela waktu untuk GRU dibentuk dari fitur di atas pada tahap Feature Engineering (Bab 5).

### 4.3 Kondisi Data

Kondisi data belum dapat disimpulkan sebelum dataset diunduh dan diperiksa. Audit data akan mencakup:

- **Missing Values:** [Diisi berdasarkan hasil pemeriksaan].
- **Duplicate Data:** [Diisi berdasarkan hasil pemeriksaan].
- **Outliers:** [Diisi berdasarkan hasil pemeriksaan].
- **Noise:** Data satelit dan reanalysis dapat memiliki ketidakpastian pengukuran atau estimasi.
- **Data Quality Issues:** Perbedaan resolusi spasial dan temporal antara FIRMS dan CAMS perlu ditangani saat integrasi. Selain itu, model CAMS EAC4 menggunakan emisi kebakaran dari GFAS yang diturunkan dari pengamatan satelit (Fire Radiative Power), sehingga fitur kebakaran dan target PM2.5 tidak sepenuhnya independen. Hal ini dicantumkan sebagai keterbatasan dalam menafsirkan hasil.
- **Asumsi Data:** Data memiliki ketergantungan spasial dan temporal. Pembagian data berdasarkan waktu akan dipertimbangkan untuk menghindari kebocoran data.

### 4.4 Exploratory Data Analysis (EDA)

EDA akan dilakukan setelah dataset berhasil dikumpulkan dan digabungkan. Visualisasi yang direncanakan meliputi:

#### Visualisasi 1 – Distribusi Konsentrasi PM2.5

**Visualisasi:** Histogram atau boxplot distribusi PM2.5.

**Insight:** [Diisi setelah visualisasi dibuat, mencakup bentuk distribusi, nilai ekstrem, dan rentang konsentrasi].

#### Visualisasi 2 – Aktivitas Hotspot dan PM2.5

**Visualisasi:** Scatter plot atau grafik agregasi jumlah hotspot/FRP terhadap PM2.5.

**Insight:** [Diisi setelah analisis dilakukan. Hubungan yang terlihat tidak langsung diartikan sebagai hubungan sebab-akibat].

#### Visualisasi 3 – Hubungan Variabel Meteorologi dan PM2.5

**Visualisasi:** Heatmap korelasi atau scatter plot suhu, kelembapan, komponen angin, dan PM2.5.

**Insight:** [Diisi berdasarkan hasil analisis aktual].

---

## 5. DATA PREPARATION [L]

### 5.1 Data Cleaning

Rencana proses data cleaning:

1. Memeriksa tipe data setiap kolom dan mengubah kolom waktu ke format datetime.
2. Memeriksa nilai kosong pada variabel penting.
3. Memeriksa dan menangani data duplikat.
4. Memeriksa nilai tidak wajar pada PM2.5, FRP, suhu, dan variabel meteorologi.
5. Memastikan satuan variabel sesuai dengan dokumentasi sumber data.
6. Memfilter data berdasarkan wilayah dan periode penelitian.
7. Memeriksa kelengkapan urutan waktu (interval tiga jam) pada setiap grid, karena fitur lag dan GRU memerlukan urutan waktu yang tidak terputus.

Strategi penanganan missing values dan outliers akan ditentukan setelah karakteristik data diketahui. Nilai tidak akan dihapus atau diimputasi tanpa mempertimbangkan alasan dan dampaknya terhadap data.

### 5.2 Feature Engineering

Fitur yang direncanakan untuk dibentuk:

- `hotspot_count`: jumlah deteksi hotspot dalam satu grid dan periode waktu.
- `avg_frp`: rata-rata FRP pada grid dan periode waktu.
- `timestamp`: waktu pengamatan yang diselaraskan; dari kolom ini diturunkan `hour` dan `month`.
- `latitude` dan `longitude`: identitas lokasi/grid.
- Variabel meteorologi yang tersedia dan relevan, seperti suhu, kelembapan, serta komponen angin.
- Fitur riwayat (lag): nilai `hotspot_count`, `avg_frp`, dan variabel meteorologi pada langkah waktu sebelumnya (rencana awal hingga t-7, setara 21 jam). Fitur ini digunakan oleh Decision Tree dan XGBoost agar memperoleh informasi historis yang setara dengan jendela waktu GRU.
- Jendela waktu (sequence window) untuk GRU: rangkaian 8 langkah waktu dari fitur yang sama pada setiap grid, dengan target PM2.5 pada langkah waktu terakhir.

Fitur hotspot akan diagregasikan ke grid spasial dan interval waktu CAMS. Penggabungan dilakukan berdasarkan lokasi dan waktu yang sesuai. Definisi grid, toleransi waktu, serta metode agregasi akan didokumentasikan setelah ditetapkan.

PM2.5 pada waktu sebelumnya tidak digunakan sebagai fitur masukan agar model mempelajari hubungan antara aktivitas kebakaran, kondisi meteorologi, dan PM2.5, bukan sekadar autokorelasi target. Jumlah lag dan panjang jendela waktu dapat disesuaikan melalui eksperimen dan akan didokumentasikan.

### 5.3 Data Transformation

Transformasi yang direncanakan:

- Mengonversi waktu ke format datetime dan menyelaraskan zona waktu sesuai metadata sumber.
- Mengonversi satuan PM2.5 ke µg/m³ jika diperlukan.
- Menggabungkan data berdasarkan grid spasial dan waktu.
- Melakukan scaling fitur numerik untuk model yang membutuhkannya, khususnya GRU.
- Membentuk rangkaian data (sliding window) per grid sebagai input GRU dengan bentuk `(jumlah_sampel, window, jumlah_fitur)`.
- Menangani nilai kategorikal jika terdapat fitur kategorikal yang digunakan.

Scaler akan di-fit hanya menggunakan data training, kemudian diterapkan pada validation dan test set untuk mencegah data leakage. Model berbasis pohon seperti Decision Tree dan XGBoost umumnya tidak membutuhkan scaling fitur.

### 5.4 Data Splitting

Karena data memiliki dimensi waktu, strategi pembagian yang direncanakan adalah pembagian kronologis, bukan random split.

Rencana awal:
- Training set: 2018–2021.
- Validation set: 2022.
- Test set: 2023.

Pembagian tersebut bersifat rencana dan akan disesuaikan dengan periode serta kelengkapan data akhir. Test set tidak digunakan untuk tuning hyperparameter. Jika periode data berubah, batas pembagian akan diperbarui dan didokumentasikan. Fitur lag dan jendela waktu hanya menggunakan langkah waktu sebelumnya dan tidak melintasi batas antarperiode pembagian.

### 5.5 Data Balancing

Data balancing seperti SMOTE tidak direncanakan karena proyek merupakan tugas regresi dengan target numerik kontinu. Jika distribusi target sangat tidak merata, strategi penanganan akan dipertimbangkan berdasarkan hasil analisis data dan kebutuhan eksperimen.

### 5.6 Ringkasan Data Preparation

| Proses | Apa yang dilakukan | Mengapa penting |
|---|---|---|
| Data cleaning | Memeriksa missing values, duplikasi, nilai tidak wajar, dan tipe data | Menjaga kualitas data sebelum pemodelan |
| Data integration | Menyelaraskan FIRMS dan CAMS berdasarkan lokasi serta waktu | Membentuk satu dataset dengan fitur dan target yang sesuai |
| Feature engineering | Membentuk jumlah hotspot, rata-rata FRP, fitur waktu, dan fitur riwayat (lag) per grid/periode | Merepresentasikan aktivitas kebakaran dan konteks temporal dalam bentuk fitur tabular |
| Transformation | Konversi satuan, format waktu, scaling bila diperlukan, dan pembentukan jendela waktu untuk GRU | Menyesuaikan data dengan kebutuhan model |
| Data splitting | Membagi data secara kronologis | Menguji kemampuan model pada periode yang lebih baru dan mengurangi data leakage |

---

---

**Catatan pengisian:** Dokumen ini merupakan draf laporan berdasarkan rancangan proyek dan README, mencakup Bab 1–5 dan telah disesuaikan dengan revisi tiga model (Decision Tree Regressor, XGBoost Regressor, dan GRU). Semua bagian bertanda `[ ... ]` atau instruksi pengisian perlu diperbarui berdasarkan informasi resmi mata kuliah dan hasil eksperimen yang benar-benar diperoleh. Jangan mengisi jumlah sampel, hasil metrik, waktu training, atau temuan EDA sebelum proses tersebut dilakukan.
