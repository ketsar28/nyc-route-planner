# -*- coding: utf-8 -*-
"""
Dibuat pada Rab Okt 01 14:00:00 2025

@author: KETSAR
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import warnings
import datetime
import shap
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, mean_squared_error, mean_squared_log_error

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

n_sample = 200_000
df_raw = pd.read_csv('train.csv', nrows=n_sample)
df_test_raw = pd.read_csv('test.csv').copy()

df_2 = df_raw[(df_raw.trip_duration > 60) & (df_raw.trip_duration < 3600 * 6)].copy()

df_2.info()

df_2['pickup_datetime'] = pd.to_datetime(df_2.pickup_datetime)
df_2['pickup_hour'] = df_2.pickup_datetime.dt.hour
df_2['pickup_day_of_week'] = df_2.pickup_datetime.dt.dayofweek

df_test_raw['pickup_datetime'] = pd.to_datetime(df_test_raw.pickup_datetime)
df_test_raw['pickup_hour'] = df_test_raw.pickup_datetime.dt.hour
df_test_raw['pickup_day_of_week'] = df_test_raw.pickup_datetime.dt.dayofweek

R = 6371
lon1,lat1,lon2,lat2 = map(np.radians, [df_2.pickup_longitude, df_2.pickup_latitude, 
                                       df_2.dropoff_longitude, df_2.dropoff_latitude])

lon1_test, lat1_test, lon2_test, lat2_test = map(np.radians, [df_test_raw.pickup_longitude, df_test_raw.pickup_latitude, df_test_raw.dropoff_longitude, df_test_raw.dropoff_latitude])

dlon = lon2-lon1
dlat = lat2-lat1
dlon_test = lon2_test - lon1_test
dlat_test = lat2_test - lat1_test


a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
df_2['distance_km'] = R * 2 * np.arcsin(np.sqrt(a)) # rasio --> sudut

a_test = np.sin(dlat_test/2.0)**2 + np.cos(lat1_test) * np.cos(lat2_test) * np.sin(dlon_test/2.0)**2
df_test_raw['distance_km'] = R * 2 * np.arcsin(np.sqrt(a_test))

y_dir = np.sin(dlon) * np.cos(lat2)
X_dir = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

y_dir_test = np.sin(dlon_test) * np.cos(lat2_test)
X_dir_test = np.cos(lat1_test) * np.sin(lat2_test) - np.sin(lat1_test) * np.cos(lat2_test) * np.cos(dlon_test)

df_2['bearing'] = np.degrees(np.arctan2(y_dir, X_dir))
df_test_raw['bearing'] = np.degrees(np.arctan2(y_dir_test, X_dir_test))

coords = np.vstack((df_2[['pickup_latitude', 'pickup_longitude']],
                    df_2[['dropoff_latitude', 'dropoff_longitude']]))

kmeans_model = MiniBatchKMeans(n_clusters=15, batch_size=256, random_state=42, n_init='auto')
kmeans_model.fit(coords)

df_2['pickup_cluster'] = kmeans_model.predict(df_2[['pickup_latitude', 'pickup_longitude']])
df_2['dropoff_cluster'] = kmeans_model.predict(df_2[['dropoff_latitude', 'dropoff_longitude']])

df_test_raw['pickup_cluster'] = kmeans_model.predict(df_test_raw[['pickup_latitude', 'pickup_longitude']])
df_test_raw['dropoff_cluster'] = kmeans_model.predict(df_test_raw[['dropoff_latitude', 'dropoff_longitude']])

df_2['route_cluster'] = df_2['pickup_cluster'].astype(str) + '_' + df_2['dropoff_cluster'].astype(str)
df_test_raw['route_cluster'] = df_test_raw['pickup_cluster'].astype(str) + '_' + df_test_raw['dropoff_cluster'].astype(str)

le = LabelEncoder()
df_2['route_cluster_encoded'] = le.fit_transform(df_2.route_cluster)
df_test_raw['route_cluster_encoded'] = le.fit_transform(df_test_raw.route_cluster)

print(df_2[['route_cluster', 'route_cluster_encoded']].head())

final_features = [
        'vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude', 'pickup_hour', 'pickup_day_of_week',
        'distance_km', 'bearing', 'pickup_cluster', 'dropoff_cluster', 'route_cluster_encoded'
    ]

X = df_2[final_features]
y = np.log1p(df_2.trip_duration)
X_test_final = df_test_raw[final_features]

print("Data sudah siap.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
for i in [X_train, X_test, y_train, y_test]:
    i.reset_index(drop=True, inplace=True)
    
print("Data training & testing sudah siap.")


nyc_bounds = {
        'min_lon' : -74.05, 'max_lon': -73.75,
        'min_lat': 40.6, 'max_lat': 40.9
    }

cluster_centers = kmeans_model.cluster_centers_
print(cluster_centers)

plt.figure(figsize=(12,12))
sns.scatterplot(x='pickup_longitude', y='pickup_latitude', data=df_2, s=1, alpha=1, color='blue')

plt.scatter(cluster_centers[:, 1], cluster_centers[:, 0], s=50, c='red', marker='X', label='centroid')

for i, centroid in enumerate(cluster_centers):
    plt.text(centroid[1], centroid[0], str(i), fontsize=5, c='black', weight='bold')
    
plt.title('peta persebaran titik penjemputan dengan centroid')
plt.xlabel('Bujur (Longitude)')
plt.ylabel('Lintang (Latitude)')
plt.xlim(nyc_bounds['min_lon'], nyc_bounds['max_lon'])
plt.ylim(nyc_bounds['min_lat'], nyc_bounds['max_lat'])
plt.show()

plt.figure(figsize=(12,12))
sns.boxplot(x='pickup_cluster', data=df_2, y=y)
plt.title('Distibusi log(durasi perjalanan) berdasarkan zona penjemputan (cluster)')
plt.xlabel('ID cluster penjemputan')
plt.ylabel('Log (trip_duration + 1)')
plt.show()


models = {
        "RandomForest": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
        "Lasso": Lasso(random_state=42),
        "Ridge": Ridge(random_state=42),
        "LinearRegresion": LinearRegression(),
        "XGBoost":XGBRegressor(n_jobs=-1, random_state=42),
        "LightGBM":LGBMRegressor(n_jobs=-1, random_state=42),
        "MLP": MLPRegressor(max_iter=300, hidden_layer_sizes=(32,16), random_state=42)
    }

scaling_need = ['MLP', 'Lasso', 'Ridge', 'LinearRegression']
result_list = []

print("\nMemulai proses perbandingan model dengan Cross-Validation...")

for name, model in models.items():
    print(f"  - Mengevaluasi: {name}")
    
    if name in scaling_need:
        print(f'scalng_need = {name}')
        eval_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
    else:
        eval_pipeline = model
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring_metrics = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    
    cv_results = cross_validate(eval_pipeline, X, y, cv=kf, scoring=scoring_metrics, n_jobs=-1)
    
    r2_mean = np.mean(cv_results['test_r2'])
    rmsle_mean = np.sqrt(-np.mean(cv_results['test_neg_mean_squared_error']))
    mae_mean = -np.mean(cv_results['test_neg_mean_absolute_error'])
    mse_mean = -np.mean(cv_results['test_neg_mean_squared_error'])
    fit_time = np.mean(cv_results['fit_time'])
    
    result_list.append({
            'Model': name,
            'R2_mean': r2_mean,
            'RMSLE_mean': rmsle_mean,
            'MAE_log_mean': mae_mean,
            'MSE_log_mean':mse_mean,
            'Fit_time_s': fit_time
        })
    
print("Proses perbandingan model selesai.")

df_results = pd.DataFrame(result_list).sort_values(by='R2_mean', ascending=False).reset_index(drop=True)

print('\n' + '='*80)
print("HASIL PERBANDINGAN PERFORMA MODEL (PENDEKATAN FUNGSI)")
print('='*80)
print(df_results)

print("\n--- Kesimpulan Awal ---")
top_pipeline = df_results.iloc[0]
best_model_name = top_pipeline['Model']
best_r2_score = top_pipeline['R2_mean']
best_rmsle_score = top_pipeline['RMSLE_mean']
best_mae_score = top_pipeline['MAE_log_mean']
best_mse_score = top_pipeline['MSE_log_mean']
best_fit_time = top_pipeline['Fit_time_s']
print(f'''
      model_name : {best_model_name}
      best_r2_score : {best_r2_score:.2f}
      best_rmsle_score : {best_rmsle_score:.2f}
      best_mae_score : {best_mae_score:.2f}
      best_mse_score : {best_mse_score:.2f}
      best_fit_time : {best_fit_time:.2f}
      ''')


models_tune = {
        'XGBoost':{
                'model': XGBRegressor(random_state=42, n_jobs=-1),
                'params': {
                        'n_estimators': [100,200,300,500],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'max_depth':[3,5,7,9],
                        'colsample_bytree': [0.7, 0.8, 0.9],
                        'subsample': [0.7,0.8,0.9]
                    }
            },
        'LightGBM':{
                'model' : LGBMRegressor(random_state=42, n_jobs=-1),
                'params': {
                        'n_estimators':[100,200,300,500],
                        'learning_rate':[0.01, 0.05, 0.1],
                        'max_depth': [-1, 5, 10],
                        'num_leaves': [20, 31, 40],
                        'colsample_bytree': [0.7, 0.8, 0.9],
                        'subsample': [0.7,0.8,0.9]
                    }
            }
    }

tuning_results = []

for name, config in models_tune.items():
    print(f'\n--- Tuning = {name} ---')
    
    random_search = RandomizedSearchCV(config['model'], 
                               config['params'],
                               n_jobs=-1,
                               verbose=1,
                               n_iter=10,
                               random_state=42,
                               scoring='r2',
                               cv=3)
    
    random_search.fit(X_train, y_train)
    
    tuning_results.append({
            'Model': name,
            'Best_score_r2': random_search.best_score_,
            'Best_params': random_search.best_params_
        })
    
print('\n--- Hasil hyperparameter tuning komparatif ---')
df_results = pd.DataFrame(tuning_results).sort_values(by='Best_score_r2', ascending=False)
print(df_results)

top_model = df_results.iloc[0]
best_model_name = top_model['Model']
best_score_r2 = top_model['Best_score_r2']
best_params = top_model['Best_params']

print(f'\nModel terbaik = {best_model_name} ({best_score_r2})')
print('--- Melatih model pilihan pada data training ---')

if best_model_name == 'XGBoost':
    tune_model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
elif best_model_name == 'LightGBM':
    tune_model = LGBMRegressor(**best_params, random_state=42, n_jobs=-1)

tune_model.fit(X_train, y_train)

print('\n Evaluasi Performa Model (X_test)')
y_pred = tune_model.predict(X_test)
y_pred[y_pred < 0] = 0

y_test_ori = np.expm1(y_test)
y_pred_ori = np.expm1(y_pred)

final_rmsle = np.sqrt(mean_squared_log_error(y_test_ori, y_pred_ori))
final_r2 = r2_score(y_test_ori, y_pred_ori)
final_mae = mean_absolute_error(y_test_ori, y_pred_ori)
final_mse = mean_squared_error(y_test_ori, y_pred_ori)

print(f'final_rmsle = {final_rmsle:.2f}')
print(f'final_r2 = {final_r2:.2f}')
print(f'final_mae = {final_mae:.2f} detik')
print(f'final_mse = {final_mse:.2f}')

print("--- Membuat Prediksi & Submission ---")

tune_model.fit(X, y)

y_pred_final = tune_model.predict(X_test_final)
y_pred_backlog = np.expm1(y_pred_final)
y_pred_backlog[y_pred_backlog < 0] = 0

result_final = {
        'id' : df_test_raw['id'],
        'trip_duration': y_pred_backlog
    }

y_pred_log = tune_model.predict(X_test)
residuals = y_test - y_pred_log

df_error_analysis = X_test.copy()
df_error_analysis['log_trip_duration_actual'] = y_test
df_error_analysis['log_trip_duration_pred'] = y_pred_log
df_error_analysis['error'] = residuals

plt.figure(figsize=(12,6))
sns.scatterplot(x='distance_km', y='error', data=df_error_analysis, alpha=0.5)
plt.title('error prediksi vs jarak perjalanan')
plt.xlabel('jarak perjalanan (km/jam)')
plt.ylabel('error (actual - prediction)')
plt.axhline(0, c='red', linestyle='--')
plt.show()

importances = tune_model.feature_importances_

df_importances = pd.DataFrame({
        'feature' : final_features,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(data=df_importances, x='importance', y='feature')
plt.xlabel('importance score')
plt.ylabel('feature name')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))
sns.scatterplot(y=y_test_ori, x=y_pred_ori, color='blue', alpha=0.5)
plt.plot([y_test_ori.min(), y_test_ori.max()], [y_test_ori.min(), y_test_ori.max()],'k--' ,lw=2)
plt.xlabel('hasil prediksi')
plt.ylabel('hasil asli')
plt.show()

explainer = shap.TreeExplainer(tune_model)
shap_values = explainer.shap_values(X.sample(5_000, random_state=42))
shap.summary_plot(shap_values, X.sample(5_000, random_state=42), plot_type='bar', show=False)
shap.summary_plot(shap_values, X.sample(5_000, random_state=42))

print('\nSHP Dependence Plot untuk distance_km')
plt.title('Dependence plot : pengaruh jarak terhadap prediksi durasi')
shap.dependence_plot('distance_km', shap_values, X.sample(5_000, random_state=41), interaction_index='pickup_hour', show=False)
plt.ylabel('shap value (dampak pada log(trip duration)')
plt.show()

# Tentukan nama file
model_filename = f'{best_model_name}_model.pkl'
kmeans_filename = 'kmeans_model.pkl'

with open(model_filename, 'wb') as f: 
    pickle.dump(tune_model, f)
    print(f'model {best_model_name} berhasil disimpan ke dalam {model_filename}')

with open(kmeans_filename, 'wb') as f: 
    pickle.dump(kmeans_model, f)
    print(f'model K-Means berhasil disimpan kedalam {kmeans_filename}')
    

with open(model_filename, 'rb') as f: 
    loaded_model = pickle.load(f)
    print(f'model {best_model_name} berhasil diload.')

with open(kmeans_filename, 'rb') as f: 
    loaded_kmeans = pickle.load(f)
    print(f'model K-Means berhasil diload.')

sample_data = X.head(1).copy()
print('data sample yang mau diprediksi')
print(sample_data)

ori_pred_log = tune_model.predict(sample_data)
ori_pred_sec = np.expm1(ori_pred_log)

loaded_pred_log = loaded_model.predict(sample_data)
loaded_pred_sec = loaded_kmeans.predict(sample_data)

print(f'prediksi dari model asli : {ori_pred_sec[0]:.2f} detik')
print(f'prediksi dari model yang dimuat : {loaded_pred_sec[0]:.2f} detik')


file_name = 'submission_final_tune.csv'
df_final = pd.DataFrame(result_final)
df_final.to_csv(file_name, index=False)
print(f'\nFile {file_name} berhasil dibuat...')
print(df_final)







