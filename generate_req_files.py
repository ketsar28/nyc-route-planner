# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 23:19:07 2025

@author: KETSAR
"""

# simpan sebagai prepare_encoder.py dan jalankan
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import joblib

print("Memuat model kmeans...")
kmeans_model = joblib.load('kmeans_model.pkl') 

print("Memuat train.csv (hanya 100rb baris)...")
df_train_raw = pd.read_csv('train.csv', nrows=100000)
df_train_cleaned = df_train_raw[(df_train_raw.trip_duration > 60) & (df_train_raw.trip_duration < 3600 * 6)].copy()

print("Memuat test.csv ukuran penuh...")
df_test_full = pd.read_csv('test.csv')

median_lat = df_test_full['pickup_latitude'].median()
median_lon = df_test_full['pickup_longitude'].median()

print(f"Nilai Median Latitude: {median_lat}")
print(f"Nilai Median Longitude: {median_lon}")
print("\n--> Salin nilai-nilai ini ke dalam kode Streamlit Anda!\n")

print("Membuat pool data pelanggan yang ringkas...")
df_pool = df_test_full[['pickup_latitude', 'pickup_longitude']].sample(20000, random_state=42)
df_pool = df_pool.rename(columns={'pickup_latitude': 'latitude', 'pickup_longitude': 'longitude'})

print("Membuat cluster rute...")
train_pickup_cluster = kmeans_model.predict(df_train_cleaned[['pickup_latitude', 'pickup_longitude']])
train_dropoff_cluster = kmeans_model.predict(df_train_cleaned[['dropoff_latitude', 'dropoff_longitude']])
train_route_cluster = [f'{p}_{d}' for p, d in zip(train_pickup_cluster, train_dropoff_cluster)]

print("Melatih dan menyimpan LabelEncoder...")
label_encoder = LabelEncoder()
label_encoder.fit(train_route_cluster)

label_filename ='label_encoder.pkl'
with open(label_filename, 'wb') as f: 
    pickle.dump(label_encoder, f)
    print(f'model label_filename berhasil disimpan kedalam {label_filename}')

df_pool.to_parquet('customer_pool.parquet')

print("Selesai! File 'label_encoder.pkl' telah dibuat.")
print("Selesai! File 'customer_pool.parquet' telah dibuat. Ukurannya sangat kecil.")






