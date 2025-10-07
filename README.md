# 🚕 NYC Intelligent Route Planner: Prediksi Durasi & Optimisasi Rute

![Demo Aplikasi](https://i.imgur.com/vHq4c6t.png)

*Catatan: Ganti gambar di atas dengan screenshot atau GIF dari aplikasi Anda agar lebih menarik.*

Ini adalah purwarupa aplikasi web yang dibangun dengan Streamlit untuk memecahkan masalah logistik perkotaan. Aplikasi ini menggunakan pendekatan dua tahap: **prediksi durasi perjalanan** dengan *machine learning* dan **optimisasi rute kendaraan** (VRP) untuk menemukan jalur paling efisien.

---

## ✨ Fitur Utama

-   🔮 **Kalkulator Durasi Perjalanan**: Memprediksi durasi perjalanan antara dua titik koordinat di NYC dengan memperhitungkan jam dan hari.
-   🔍 **Interpretasi Model (SHAP)**: Menampilkan visualisasi faktor-faktor apa saja (seperti jarak, jam sibuk, atau lokasi) yang paling mempengaruhi hasil prediksi durasi untuk setiap perjalanan.
-   🗺️ **Perencana Rute Optimal (VRP)**: Mensimulasikan skenario *Vehicle Routing Problem* untuk armada kendaraan, menghitung, dan memvisualisasikan rute paling efisien di peta interaktif.
-   ℹ️ **Informasi Proyek**: Halaman terdedikasi yang menjelaskan latar belakang, teknologi, dan sumber data yang digunakan dalam proyek.
-   🎨 **UI Modern & Gelap**: Antarmuka yang bersih dan modern dengan tema gelap yang nyaman di mata.

---

## 🎯 Masalah yang Diselesaikan

Perusahaan di bidang logistik dan mobilitas perkotaan (seperti taksi, kurir, atau layanan pengiriman) seringkali menghadapi inefisiensi operasional yang disebabkan oleh ketidakpastian lalu lintas di kota besar. Perencanaan rute yang tidak optimal mengakibatkan:

-   **Pemborosan Biaya**: Peningkatan konsumsi bahan bakar dan biaya lembur pengemudi.
-   **Produktivitas Rendah**: Jumlah pengiriman atau perjalanan yang bisa diselesaikan per hari lebih sedikit.
-   **Ketidakpuasan Pelanggan**: Estimasi waktu tiba (ETA) yang tidak akurat dan keterlambatan.

Proyek ini bertujuan untuk menyediakan sebuah alat bantu pengambilan keputusan cerdas untuk mengatasi tantangan tersebut.

---

## 🛠️ Solusi yang Dibangun

Solusi ini menggunakan pendekatan dua tahap yang saling terhubung:

1.  **Tahap 1: Model Prediksi (Machine Learning)**
    -   Sebuah model **XGBoost** dilatih menggunakan data historis jutaan perjalanan taksi di NYC.
    -   Model ini mempelajari pola kompleks antara fitur perjalanan (jarak, lokasi, waktu) dan durasi perjalanannya. Hasilnya adalah sebuah "mesin prediksi" yang akurat.

2.  **Tahap 2: Model Optimisasi (Heuristik)**
    -   Model prediksi dari tahap 1 digunakan sebagai "mesin penghitung biaya (waktu)" untuk setiap kemungkinan segmen rute dari satu pelanggan ke pelanggan lainnya.
    -   Library **Google OR-Tools** kemudian digunakan untuk menyelesaikan *Vehicle Routing Problem* (VRP). *Solver* ini secara cerdas mencari kombinasi rute terbaik untuk seluruh armada dengan tujuan meminimalkan total waktu perjalanan.

---

## 🚀 Teknologi yang Digunakan

-   **Bahasa Pemrograman**: Python
-   **Analisis & Manipulasi Data**: Pandas, NumPy
-   **Machine Learning**: Scikit-learn, XGBoost
-   **Interpretasi Model**: SHAP, Matplotlib
-   **Optimisasi Rute**: Google OR-Tools
-   **Dashboard & Visualisasi**: Streamlit, PyDeck

---
