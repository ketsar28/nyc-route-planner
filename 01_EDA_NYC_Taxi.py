# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 12:48:18 2025

@author: KETSAR
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime
from sklearn.model_selection import train_test_split
import lightgbm as lgb
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

n_sample = 100_000
chunk_size = 50_000
n_total = 1_000_000

print('mulai proses sampling data')

header = pd.read_csv('train.csv', nrows=0).columns.tolist()
chunks = []

for chunk in pd.read_csv('train.csv', chunksize=chunk_size, low_memory=False):
    print(f'chunk = {chunk} | total = {len(chunk)}')
    print(f'len(chunk)/n_total = {len(chunk)/n_total}')
    print(f'total chunk sample = {n_sample * (len(chunk)/n_total)}')
    chunks.append(chunk.sample(n=int(n_sample * (len(chunk)/n_total)), random_state=42))
    
df_sample = pd.concat(chunks, ignore_index=True)
df_sample.info()
print(df_sample.head())

df_sample.columns
missing_mean = df_sample.isnull().mean() * 100
missing_sum = df_sample.isnull().sum()

print(f'rata-rata yang hilang = \n{missing_mean}')
print(f'\njumlah yang hilang = \n{missing_sum}')

df_sample['pickup_datetime'] = pd.to_datetime(df_sample['pickup_datetime'])
df_sample['dropoff_datetime'] = pd.to_datetime(df_sample['dropoff_datetime'])

df_sample.describe()

df_sample['pickup_hour'] = df_sample['pickup_datetime'].dt.hour
df_sample['pickup_day_of_week'] = df_sample['pickup_datetime'].dt.dayofweek # Senin=0, Minggu=6
df_sample['pickup_month'] = df_sample['pickup_datetime'].dt.month

# jarak di permukaan bola/bumi
def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371  # Radius bumi dalam kilometer
    
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    distance = R * c
    return distance

# Terapkan fungsi untuk membuat kolom jarak baru
df_sample['distance_km'] = haversine_distance(df_sample['pickup_longitude'], df_sample['pickup_latitude'],
                                     df_sample['dropoff_longitude'], df_sample['dropoff_latitude'])

print("\n--- Dataset setelah Feature Engineering ---")
print(df_sample.head())


print("\nMembuat visualisasi EDA...")

# 1. Distribusi Durasi Perjalanan (Target Variable)
plt.figure(figsize=(12, 6))
sns.histplot(df_sample['trip_duration'], bins=100)
plt.title('Distribusi Durasi Perjalanan (Detik)')
plt.xlabel('Durasi (detik)')
plt.ylabel('Frekuensi')
plt.show()

# Coba gunakan log transform untuk melihat distribusinya lebih jelas.
plt.figure(figsize=(12, 6))
sns.histplot(np.log1p(df_sample['trip_duration']), bins=100) # log1p(x) = log(1+x), aman untuk nilai 0
plt.title('Distribusi Log dari Durasi Perjalanan')
plt.xlabel('Log(Durasi + 1)')
plt.ylabel('Frekuensi')
plt.show()


# 2. Durasi Perjalanan rata-rata berdasarkan Jam Keberangkatan
plt.figure(figsize=(12, 6))
df_sample.groupby('pickup_hour')['trip_duration'].mean().plot(kind='bar')
plt.title('Rata-rata Durasi Perjalanan vs Jam Keberangkatan')
plt.xlabel('Jam Keberangkatan')
plt.ylabel('Rata-rata Durasi (detik)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.show()


# 3. Durasi Perjalanan vs Jarak
plt.figure(figsize=(10, 8))
sns.scatterplot(x='distance_km', y='trip_duration', data=df_sample, alpha=0.5)
plt.title('Durasi Perjalanan vs Jarak')
plt.xlabel('Jarak (km)')
plt.ylabel('Durasi (detik)')
plt.xlim(0, 50) # Batasi sumbu x untuk menghilangkan outlier jarak
plt.ylim(0, 10000) # Batasi sumbu y untuk menghilangkan outlier durasi
plt.show()
# Seharusnya ada korelasi positif, tapi dengan banyak variasi.

print("\nProses EDA awal selesai.")

durasi_1 = 61
durasi_2 = 12526

td_1 = datetime.timedelta(seconds=durasi_1)
td_2 = datetime.timedelta(seconds=durasi_2)

print(f"{durasi_1} detik = {td_1}")
print(f"{durasi_2} detik = {td_2}")

def second_conversion(second):
    second=int(second)
    
    hour = second // 3600
    remain_second = second % 3600
    minute = remain_second // 60
    end_second = remain_second % 60
    
    return f'{hour} jam {minute} menit {end_second} detik'

print(f"{durasi_1} detik = {second_conversion(durasi_1)}")
print(f"{durasi_2} detik = {second_conversion(durasi_2)}")

df_sample['duration_hour'] = df_sample['trip_duration'] / 3600
# Ganti durasi 0 jam dengan nilai kecil untuk menghindari error
df_sample['speed_kmh'] = df_sample['distance_km'] / df_sample['duration_hour'].replace(0, 0.00001)

print(df_sample['speed_kmh'].describe())

# Visualisasi distribusi kecepatan
plt.figure(figsize=(12, 6))
sns.histplot(df_sample['speed_kmh'], bins=100)
plt.title('Distribusi Kecepatan Perjalanan (km/jam)')
plt.xlabel('Kecepatan (km/jam)')
plt.xlim(0, 100) # Fokus pada kecepatan yang wajar
plt.show()

# Contoh membersihkan outlier
# Misal, kita anggap durasi wajar antara 1 menit hingga 6 jam, dan jarak > 0
df_cleaned = df_sample[
    (df_sample['trip_duration'] > 60) & 
    (df_sample['trip_duration'] < 3600 * 6) &
    (df_sample['distance_km'] > 0)
].copy()

# Vendor ID vs Durasi
plt.figure(figsize=(8, 6))
sns.boxplot(x='vendor_id', y=np.log1p(df_cleaned['trip_duration']), data=df_cleaned)
plt.title('Log(Durasi Perjalanan) vs Vendor ID')
plt.show()

print(np.log1p(df_cleaned['trip_duration']))

# Distribusi Jumlah Penumpang
plt.figure(figsize=(8,6))
sns.countplot(x='passenger_count', data=df_cleaned)
plt.title('Distribusi Jumlah Penumpang')
plt.show()


# Jumlah Penumpang vs Durasi
plt.figure(figsize=(8,6))
sns.boxplot(data=df_sample, x='passenger_count', y=np.log1p(df_cleaned.trip_duration))
plt.title('Log (Durasi Perjalanan) vs Jumlah Penumpang')
plt.show()

nyc_bounds = {
        'min_lon' : -74.05, 'max_lon': -73.75,
        'min_lat': 40.6, 'max_lat': 40.9
    }


df_map = df_cleaned[
        (df_cleaned['pickup_longitude'].between(nyc_bounds['min_lon'], nyc_bounds['max_lon'])) &
        (df_cleaned['pickup_latitude'].between(nyc_bounds['min_lat'], nyc_bounds['max_lat'])) &
        (df_cleaned['dropoff_longitude'].between(nyc_bounds['min_lon'], nyc_bounds['max_lon'])) &
        (df_cleaned['dropoff_latitude'].between(nyc_bounds['min_lat'], nyc_bounds['max_lat']))
    ].copy()


# Plot lokasi penjemputan
plt.figure(figsize=(12,12))
sns.scatterplot(x='pickup_longitude', y='pickup_latitude', data=df_map, s=2, alpha=0.2, color='blue')
plt.title('Map Kepadatan Lokasi Penjemputan di NYC')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

not_importance = ['vendor_id', 'pickup_datetime', 'dropoff_datetime', 'duration_hour']
numeric_cols = [col for col in df_cleaned.columns if df_cleaned[col].dtype !='object' and col not in not_importance]

corr_mtx = df_cleaned[numeric_cols].corr()

plt.figure(figsize=(14,10))
sns.heatmap(corr_mtx, annot=True, cmap='coolwarm', fmt='.2f', linewidth=0.5)
plt.title('Heatmap Korelasi Antar Fitur Numerik')
plt.show()


def remove_corr_feature(df, threshold):
    corr_featname = set()
    corr_mtx = df.corr()
    for col in range(len(corr_mtx.columns)):
        for row in range(col):
            if abs(corr_mtx.iloc[col, row] > threshold):
                colname = corr_mtx.columns[col]
                corr_featname.add(colname)
    return list(corr_featname)

features_to_drop = remove_corr_feature(df_cleaned[numeric_cols], 0.8)
df_no_corr = df_cleaned.drop(columns=features_to_drop)


final_features = ['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'pickup_hour', 'pickup_day_of_week', 'pickup_month', 'distance_km']

X = df_cleaned[final_features]
y = np.log1p(df_cleaned['trip_duration'])





































