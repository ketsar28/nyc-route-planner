# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 12:51:07 2025

@author: KETSAR
"""

import pandas as pd
import warnings
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from ortools.linear_solver import pywraplp
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# load model
print("*"*20,'load model prediksi & preprocessing',"*"*20)
try:
    final_model = joblib.load('XGBoost_model.pkl')
    kmeans_model = joblib.load('kmeans_model.pkl')
    print('model berhasil di load')
except FileNotFoundError:
    print('error: pastikan file berada di direktori yang sama')
    exit()
    
    
df_train_raw = pd.read_csv('train.csv', nrows=100000) 
df_train_cleaned = df_train_raw[(df_train_raw.trip_duration > 60) & (df_train_raw.trip_duration < 3600 * 6)].copy()
train_pickup_cluster = kmeans_model.predict(df_train_cleaned[['pickup_latitude', 'pickup_longitude']])
train_dropoff_cluster = kmeans_model.predict(df_train_cleaned[['dropoff_latitude', 'dropoff_longitude']])
train_route_cluster = [f"{p}_{d}" for p, d in zip(train_pickup_cluster, train_dropoff_cluster)]

le = LabelEncoder()
le.fit(train_route_cluster)

print('*'*20,'preparing skenario optimisasi','*'*20)
df_test_op_raw = pd.read_csv('test.csv')

# definisi parameter
NUM_VEHICLES = 4
NUM_CUSTOMERS = 50
DEPOT_INDEX = 0

depot_coords = kmeans_model.cluster_centers_[7]
depot_options_2 = {i: f"Zona Depot {i}" for i in range(len(kmeans_model.cluster_centers_))}

customer_location = df_test_op_raw.sample(NUM_CUSTOMERS, random_state=42)

df_locations_2 = pd.DataFrame([{'latitude': depot_coords[0], 'longitude': depot_coords[1], 'name': 'Depot'}])
df_customer_coords_2 = customer_location[['pickup_latitude', 'pickup_longitude']].rename(columns={'pickup_latitude':'latitude', 'pickup_longitude':'longitude'})

df_customer_coords_2['name'] = [f'Pelanggan {i+1}' for i in range(len(df_customer_coords_2))]
df_locations_2 = pd.concat([df_locations_2, df_customer_coords_2], ignore_index=True)



data = {
    'latitude' : depot_coords[0],
    'longitude' : depot_coords[1]
}
df_location = pd.DataFrame([data])
df_location

df_customer_coords = customer_location[['pickup_latitude', 'pickup_longitude']].rename(columns={
        'pickup_latitude': 'latitude',
        'pickup_longitude': 'longitude'
    })
df_location = pd.concat([df_location, df_customer_coords], ignore_index=True)
print(f'skenario dibuat 1 depot, {NUM_CUSTOMERS} pelanggan, dan {NUM_VEHICLES} kendaraan')

def predict_travel_time(start_lat, start_lon, end_lat, end_lon, hour, day_of_week):
    
    R = 6371
    lon1, lat1, lon2, lat2 = map(np.radians, [start_lon, start_lat, end_lon, end_lat])
    dlon = lon2-lon1
    dlat = lat2-lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    distance_km = R * 2 * np.arcsin(np.sqrt(a))
    
    y_bearing = np.sin(dlon) * np.cos(lat2)
    x_bearing = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(y_bearing, x_bearing))
    
    pickup_cluster = kmeans_model.predict(np.array([[start_lat, start_lon]]))[0]
    dropoff_cluster = kmeans_model.predict(np.array([[end_lat, end_lon]]))[0]
    
    route_cluster_str = f"{pickup_cluster}_{dropoff_cluster}"
    try:
        route_cluster_encoded = le.transform([route_cluster_str])[0]
    except ValueError:
        route_cluster_encoded = -1
    
    features_df = pd.DataFrame([{
        'vendor_id': 1, 
        'passenger_count': 1, 
        'pickup_longitude': start_lon, 
        'pickup_latitude': start_lat,
        'dropoff_longitude': end_lon, 
        'dropoff_latitude': end_lat,
        'pickup_hour': hour, 
        'pickup_day_of_week': day_of_week,
        'distance_km': distance_km, 
        'bearing': bearing,
        'pickup_cluster': pickup_cluster, 
        'dropoff_cluster': dropoff_cluster,
        'route_cluster_encoded': route_cluster_encoded
    }])
    
    final_features = [
        'vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude', 'pickup_hour', 'pickup_day_of_week',
        'distance_km', 'bearing', 'pickup_cluster', 'dropoff_cluster', 'route_cluster_encoded'
    ]
    
    pred_log = final_model.predict(features_df[final_features])
    return int(np.expm1(pred_log)[0])
    

print("\nMembuat matriks waktu perjalanan (cost matrix)... Ini mungkin butuh waktu.")
num_locations = len(df_location)
time_matrix = np.zeros((num_locations, num_locations), dtype=int)

start_hour = 9
start_day = 1

for from_node in range(num_locations):
    for to_node in range(num_locations):
        if from_node == to_node:
            continue
        start_lat, start_lon = df_location.iloc[from_node][['latitude','longitude']]
        end_lat, end_lon = df_location.iloc[to_node][['latitude', 'longitude']]
        travel_time = predict_travel_time(start_lat, start_lon, end_lat, end_lon, start_hour, start_day)
        time_matrix[from_node, to_node] = travel_time
        

print('matriks waktu sudah selesai dibuat')

manager = pywrapcp.RoutingIndexManager(num_locations, NUM_VEHICLES, DEPOT_INDEX)
routing = pywrapcp.RoutingModel(manager)

def time_callback(from_idx, to_idx):
    from_node = manager.IndexToNode(from_idx)
    to_node = manager.IndexToNode(to_idx)
    return time_matrix[from_node, to_node]

transit_callback_idx = routing.RegisterTransitCallback(time_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_idx)

routing.AddDimension(transit_callback_idx, 0, 28800, True, 'Time')
time_dimension = routing.GetDimensionOrDie('Time')
time_dimension.SetGlobalSpanCostCoefficient(100)

search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
solution = routing.SolveWithParameters(search_parameters)

def print_solution(manager, routing, solution, df_location):
    print(f'tujuan (total waktu) = {solution.ObjectiveValue()} detik')
    total_time=0
    route_times = []
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14,14))
    colors = plt.cm.get_cmap('gist_rainbow', NUM_VEHICLES)
    
    for vehicle_id in range(manager.GetNumberOfVehicles()):
        idx = routing.Start(vehicle_id)
        plan_output = f'rute untuk kendaraan {vehicle_id} :\n'
        route_time = 0
        route_node = []
        
        while not routing.IsEnd(idx):
            node_idx = manager.IndexToNode(idx)
            route_node.append(node_idx)
            previous_idx = idx
            idx = solution.Value(routing.NextVar(idx))
            route_time += routing.GetArcCostForVehicle(previous_idx, idx, vehicle_id)
            plan_output += f'  {node_idx} -->'
            
        node_idx = manager.IndexToNode(idx)
        route_node.append(node_idx)
        plan_output += f'  {node_idx}\n'
        plan_output += f'Total waktu rute = {route_time} detik ({route_time/3600:.2f} jam)\n'
        print(plan_output)
        total_time += route_time
        
        if len(route_node) > 1 :
            route_lats = df_location.iloc[route_node]['latitude']
            route_lons = df_location.iloc[route_node]['longitude']
            ax.plot(route_lons, route_lats, marker='o', linestyle='-', color=colors(vehicle_id), label=f'kendaraan {vehicle_id}')
            
        print("\n" + "="*40)
        print("       ANALISIS BIAYA (COST ANALYSIS)")
        print("="*40)
        total_penalty = solution.ObjectiveValue() - total_time
        print(f'Total Waktu Perjalanan (Biaya Utama) : {total_time} detik --> {total_time/3600:.2f} jam')
        print(f'Total Biaya Penalti Keseimbangan     : {total_penalty} detik --> {total_penalty/3600:.2f} jam')
        print(f'Nilai Tujuan (Objective Value) Final : {solution.ObjectiveValue()} detik (Waktu + Penalti) --> {solution.ObjectiveValue()/3600:.2f} jam')
    
        if route_times:
            min_time = min(route_times)
            max_time = max(route_times)
            span = max_time - min_time
            print(f"\nDurasi Rute Terpanjang: {max_time} detik")
            print(f"Durasi Rute Terpendek : {min_time} detik")
            print(f"Perbedaan (Span)      : {span} detik")
        print("=" * 40 + "\n")

    print(f'total waktu untuk semua rute : {total_time} detik ({total_time/3600:.2f} jam)')
        
    ax.scatter(df_location.iloc[0]['longitude'], df_location.iloc[0]['latitude'], c='white', s=200, marker='s', label='Depot')
    ax.set_title('visualisasi rute optimal', fontsize=16)
    ax.set_xlabel('longitude', fontsize=12)
    ax.set_ylabel('latitude', fontsize=12)
    ax.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()
    print(f'route_node = {route_node}')

if solution:
    print_solution(manager, routing, solution, df_location)
else:
    print('solusi engga di temukan')
        
        
        
        


























































