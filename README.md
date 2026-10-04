# MLOps - Obesity Classification 🚀

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=Prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/grafana-%23F46800.svg?style=for-the-badge&logo=grafana&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)

Selamat datang di repositori **Obesity Classification MLOps**! Proyek ini merupakan submission dari kelas Machine Learning Operations (MLOps), dengan fokus pada pengembangan model *machine learning* secara *end-to-end* untuk klasifikasi tingkat obesitas. Proyek ini mendemonstrasikan implementasi praktik-praktik MLOps yang mencakup eksperimen, *tracking*, *containerization*, serta *monitoring & logging*.

---

## 📌 Deskripsi Proyek

Proyek ini bertujuan untuk membangun model machine learning yang dapat memprediksi tingkat obesitas seseorang berdasarkan berbagai fitur kesehatan dan gaya hidup. Alih-alih hanya berfokus pada pembuatan model, repositori ini menitikberatkan pada aspek rekayasa perangkat lunak dan operasi model (MLOps).

Fitur Utama:
1. **Model Building & Experiment Tracking**: Melatih model dan melacak parameter, metrik, dan *artifacts* secara sistematis.
2. **Serving**: Membungkus model dalam format API agar mudah diintegrasikan.
3. **Containerization**: Membuat arsitektur berbasis *container* untuk konsistensi lingkungan eksekusi.
4. **Monitoring & Alerting**: Memantau performa *hardware* serta ketersediaan sistem API, lengkap dengan sistem peringatan otomatis (*alerting*).

---

## 📁 Struktur Repositori

Repositori ini terbagi menjadi dua bagian utama:

### 1. `Membangun_model/`
Folder ini berisi kode, dataset, serta *artifacts* yang berkaitan dengan proses pelatihan model dan *experiment tracking*.
- **`modelling.py` & `modelling_tuning.py`**: Script untuk pelatihan dan *hyperparameter tuning* model.
- **Dataset**: `dataset_train.csv`, `dataset_test.csv`, `obesity_classification_preprocessing.csv`.
- **Tracking & Logging**: Terintegrasi dengan **MLflow** dan **DagsHub**. Di sini Anda akan menemukan screenshot bukti jalannya *experiment tracking* (*autologging* dan *manual logging*).

### 2. `Monitoring_dan_Logging/`
Folder ini memuat konfigurasi serta script untuk men-deploy model ke dalam layanan (*serving*), beserta setup pemantauannya.
- **`inference.py` & `prometheus_exporter.py`**: Script Flask API untuk melayani prediksi dan mengekspos metrik Prometheus.
- **`Dockerfile` & `docker-compose.yml`**: Berkas untuk orkestrasi kontainer aplikasi, Prometheus, dan Grafana.
- **Monitoring Tools**: Konfigurasi `prometheus.yml` dan direktori berisi bukti pengamatan dan *alerting* via Grafana.

---

## 🛠️ Teknologi yang Digunakan

- **Bahasa Pemrograman**: Python 3.10
- **Machine Learning Library**: Scikit-Learn, Pandas, NumPy
- **Experiment Tracking**: MLflow, DagsHub
- **Web Framework**: Flask
- **Containerization**: Docker & Docker Compose
- **Monitoring & Observability**: Prometheus, Grafana

---

## 🚀 Cara Menjalankan Proyek

### 1. Pelatihan Model (Opsional)
Jika Anda ingin melatih model atau menjalankan ulang eksperimen:
```bash
cd Membangun_model
pip install -r requirements.txt
python modelling_tuning.py
```

### 2. Menjalankan Layanan (Serving & Monitoring)
Pastikan Docker dan Docker Compose telah terinstal.
```bash
cd Monitoring_dan_Logging
docker-compose up -d --build
```

Perintah di atas akan menjalankan beberapa layanan berikut:
- **Model API**: Dapat diakses di port `5001`.
- **Prometheus**: Dashboard metrik di port `9090` (secara default tergantung di docker-compose).
- **Grafana**: Visualisasi *monitoring* dapat diakses di port `3000`.

---

## 📊 Hasil dan Dokumentasi
- **Eksperimen**: Tersimpan di dalam folder `mlruns/` atau dapat dilihat pada log *DagsHub*.
- **Monitoring**: Screenshot metrik performa (CPU, RAM, Request Rate) dan sistem *alerting* telah didokumentasikan di folder `bukti monitoring Grafana/` dan `bukti alerting Grafana/`.

---

Dibuat untuk tugas/submission Dicoding MLOps oleh **M. Sohibbal**.
