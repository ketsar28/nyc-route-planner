# -*- coding: utf-8 -*-
"""
@author: KETSAR
"""

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import pickle
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="NYC Route Optimization",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Mengubah warna latar belakang utama */
    .main {
        background-color: #0E1117;
    }
    /* Mengubah warna teks header */
    h1, h2, h3 {
        color: #FFFFFF;
    }
    /* Mengubah warna teks biasa */
    .stApp {
        color: #E0E0E0;
    }
    /* Style untuk kartu metrik */
    .stMetric {
        background-color: #1E2128;
        border-radius: 10px;
        padding: 15px;
    }
    /* Style untuk Footer */
    .footer {
        width: 100%;
        text-align: center;
        padding: 20px 0px;
        color: #808080; /* Warna abu-abu */
        font-size: 14px;
        border-top: 1px solid #2e2e2e;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models(model_filename, kmeans_filename, encoder_filename):
    try:
        with open(model_filename, 'rb') as mod_file:
            pred_model = pickle.load(mod_file)
            print(f"File model '{model_filename}' berhasil diload.")
            
        with open(kmeans_filename, 'rb') as k_file:
            kmeans_model = pickle.load(k_file)
            print(f"File model '{kmeans_filename}' berhasil diload.")
            
        with open(encoder_filename, 'rb') as enc_file:
            label_encoder = pickle.load(enc_file)
        
        return pred_model, kmeans_model, label_encoder
        
    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: {e}. Pastikan file berada di working directory yang sama.")
        return None, None, None

@st.cache_resource
def load_explainer(_model):
    """Membuat dan menyimpan SHAP TreeExplainer di cache."""
    return shap.TreeExplainer(_model)

def predict_travel_time(start_lat, start_lon, end_lat, end_lon, hour, day_of_week, model, kmeans, le):
    """Fungsi untuk memprediksi durasi perjalanan tunggal."""
    R = 6371
    lon1, lat1, lon2, lat2 = map(np.radians, [start_lon, start_lat, end_lon, end_lat])
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    distance_km = R * 2 * np.arcsin(np.sqrt(a))
    
    y_bearing = np.sin(dlon) * np.cos(lat2)
    x_bearing = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(y_bearing, x_bearing))
    
    pickup_cluster = kmeans.predict(np.array([[start_lat, start_lon]]))[0]
    dropoff_cluster = kmeans.predict(np.array([[end_lat, end_lon]]))[0]
    
    route_cluster_str = f"{pickup_cluster}_{dropoff_cluster}"
    try:
        route_cluster_encoded = le.transform([route_cluster_str])[0]
    except ValueError:
        route_cluster_encoded = -1

    features_df = pd.DataFrame([{
        'vendor_id': 1, 'passenger_count': 1,
        'pickup_longitude': start_lon, 'pickup_latitude': start_lat,
        'dropoff_longitude': end_lon, 'dropoff_latitude': end_lat,
        'pickup_hour': hour, 'pickup_day_of_week': day_of_week,
        'distance_km': distance_km, 'bearing': bearing,
        'pickup_cluster': pickup_cluster, 'dropoff_cluster': dropoff_cluster,
        'route_cluster_encoded': route_cluster_encoded
    }])
    
    final_features = [
        'vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude', 'pickup_hour', 'pickup_day_of_week',
        'distance_km', 'bearing', 'pickup_cluster', 'dropoff_cluster',
        'route_cluster_encoded'
    ]
    
    pred_log = model.predict(features_df[final_features])
    return int(np.expm1(pred_log)[0])

@st.cache_data
def run_optimization(_locations_df, num_vehicles, depot_index):
    """Menjalankan seluruh proses optimisasi dan mengembalikan hasilnya."""
    num_locations = len(_locations_df)
    time_matrix = np.zeros((num_locations, num_locations), dtype=int)
    start_hour = 9; start_day = 1

    progress_bar = st.progress(0, text="Menghitung matriks waktu perjalanan...")
    for from_node in range(num_locations):
        for to_node in range(num_locations):
            if from_node == to_node: continue
            start_lat, start_lon = _locations_df.iloc[from_node][['latitude', 'longitude']]
            end_lat, end_lon = _locations_df.iloc[to_node][['latitude', 'longitude']]
            travel_time = predict_travel_time(start_lat, start_lon, end_lat, end_lon, start_hour, start_day, final_model, kmeans_model, le)
            time_matrix[from_node, to_node] = travel_time
        progress_bar.progress((from_node + 1) / num_locations, text=f"Menghitung dari lokasi {from_node+1}/{num_locations}")
    progress_bar.empty()

    manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return time_matrix[from_node, to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    routing.AddDimension(transit_callback_index, 0, 28800, True, 'Time')
    time_dimension = routing.GetDimensionOrDie('Time')
    time_dimension.SetGlobalSpanCostCoefficient(100)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        routes = []
        total_time = 0
        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route_nodes = []
            route_time = 0
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_nodes.append(node_index)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_time += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            node_index = manager.IndexToNode(index)
            route_nodes.append(node_index)
            
            if len(route_nodes) > 1:  
                total_time += route_time
                routes.append({
                    "vehicle_id": vehicle_id,
                    "nodes": route_nodes,
                    "time": route_time
                })
        return routes, total_time, solution.ObjectiveValue()
    
    return None, None, None

final_model, kmeans_model, le = load_models('XGBoost_model.pkl', 'kmeans_model.pkl', 'label_encoder.pkl')
df_test_raw = pd.read_parquet('customer_pool.parquet')
explainer = load_explainer(final_model)


st.title("🚕 NYC Intelligent Route Planner")
st.markdown("Sebuah purwarupa untuk memprediksi durasi perjalanan dan mengoptimalkan rute logistik menggunakan Machine Learning.")

tab1, tab2, tab3 = st.tabs(["🔮 Kalkulator Durasi", "🗺️ Perencana Rute Optimal", "ℹ️ Tentang Proyek"])

with tab1:
    st.header('Prediksi Waktu Tempuh Antar Titik')
    st.markdown('Prediksikan waktu tempuh antara dua titik di NYC menggunakan model XGBoost yang telah dilatih.')
    
    MEDIAN_LAT_DEFAULT = 40.7521
    MEDIAN_LON_DEFAULT = -73.9817
    
    col_1, col_2 = st.columns(2)
    with col_1:
        st.subheader('TITIK AWAL (Pickup)')
        start_lat = st.number_input('Masukkan Latitude Awal:', value=MEDIAN_LAT_DEFAULT, format="%.4f", key="start_lat")
        start_lon = st.number_input('Masukkan Longitude Awal:', value=MEDIAN_LON_DEFAULT, format='%.4f', key="start_lon")
    with col_2:
        st.subheader('TITIK AKHIR (Dropoff)')
        end_lat = st.number_input('Masukkan Latitude Akhir:', value=MEDIAN_LAT_DEFAULT, format="%.4f", key="end_lat")
        end_lon = st.number_input('Masukkan Longitude Akhir:', value=MEDIAN_LON_DEFAULT, format='%.4f', key="end_lon")
    
    st.subheader('Parameter Waktu')
    pickup_hour = st.slider('Jam Penjemputan (0-23):', 0, 23, 17)
    day_of_week_names = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
    pickup_day = st.selectbox('Hari Penjemputan:', options=range(7), format_func=lambda x: day_of_week_names[x])
    
    if st.button('Prediksi Durasinya!'):
        if final_model is not None:
            with st.spinner('Model lagi mikir dulu sebentar...'):
                duration_seconds = predict_travel_time(start_lat, start_lon, end_lat, end_lon, pickup_hour, pickup_day, final_model, kmeans_model, le)
                duration_minutes = duration_seconds / 60
                st.success('Yes! aku dapet prediksinya nih')
                st.metric(label='Estimasi Durasi Perjalanan', value=f'{duration_minutes:.1f} Menit')
                
        else:
            st.error('Maaf yaa aku ga bisa prediksi, modelnya gagal nih. Coba kamu cek lagi ya modelnya!')

with tab2:
    st.header('Simulasi Optimisasi Rute untuk Armada')
    st.markdown('Simulasikan skenario *Vehicle Routing Problem* (VRP) untuk armada kendaraan di NYC.')
    
    with st.sidebar:
        st.header('⚙️ Kontrol & Parameter Optimisasi')
        num_vehicles = st.slider('Jumlah Kendaraan:', 1, 10, 4)
        num_customers = st.slider('Jumlah Pelanggan:', 10, 100, 50)
        depot_options = {i: f"Zona Depot {i}" for i in range(len(kmeans_model.cluster_centers_))}
        depot_cluster_id = st.selectbox('Pilih Zona Depot:', options=list(depot_options.keys()), format_func=lambda x: depot_options[x], index=5)
    
    if st.button('Cari Rute Yang Optimal!'):
        if final_model is not None:
            with st.spinner(f'Aku siapin skenario buat {num_vehicles} kendaraan dan {num_customers} pelanggan...'):
                depot_coords = kmeans_model.cluster_centers_[depot_cluster_id]
                customer_locations = df_test_raw.sample(num_customers, random_state=42)
                df_depot = pd.DataFrame([{'latitude': depot_coords[0], 'longitude': depot_coords[1]}])
                df_locations = pd.concat([df_depot, customer_locations], ignore_index=True)
            
            routes, total_time, objective_value = run_optimization(df_locations, num_vehicles, 0)
            
            if routes:
                st.success('Yes! aku ketemu nih rute yang optimal')
                
                col_1, col_2, col_3 = st.columns(3)
                col_1.metric('Total Waktu Tempuh', f'{total_time/3600:.2f} Jam')
                col_2.metric('Kendaraan Digunakan', f'{len(routes)} dari {num_vehicles}')
                col_3.metric('Total Pelanggan', f'{num_customers}')

                path_layer_data = []
                customer_points_data = []
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), 
                          (255, 0, 255), (255, 128, 0), (128, 0, 255), (0, 128, 0), (128, 128, 128)]
                
                for i, route in enumerate(routes):
                    path_coords = []
                    route_color = colors[i % len(colors)]
                    for stop_number, node_idx in enumerate(route['nodes']):
                        location_info = df_locations.iloc[node_idx]
                        coords = [location_info['longitude'], location_info['latitude']]
                        path_coords.append(coords)
                        if node_idx != 0:
                            tooltip_text = (f"<b>Kendaraan {route['vehicle_id']} | Stop ke-{stop_number}</b><br>"
                                            f"Koordinat: ({coords[1]:.4f}, {coords[0]:.4f})")
                            customer_points_data.append({'coordinates': coords, 'color': route_color, 'tooltip': tooltip_text})
                    path_layer_data.append({'path': path_coords, 'color': route_color, 'tooltip': f"<b>Rute Kendaraan {route['vehicle_id']}</b>"})

                view_state = pdk.ViewState(latitude=depot_coords[0], longitude=depot_coords[1], zoom=11, pitch=45)
                
                depot_tooltip_text = (f"<b>Depot Pusat</b><br>"
                                      f"Koordinat: ({depot_coords[0]:.4f}, {depot_coords[1]:.4f})")
                depot_data = [{'coordinates': [depot_coords[1], depot_coords[0]], 'tooltip': depot_tooltip_text}]
                
                depot_layer = pdk.Layer('ScatterplotLayer', data=depot_data, pickable=True, get_position='coordinates', 
                                        get_fill_color=[255, 255, 255], get_radius=150, radius_min_pixels=7)
                customer_layer = pdk.Layer('ScatterplotLayer', data=customer_points_data, pickable=True, get_position='coordinates', 
                                           get_fill_color='color', get_radius=100, radius_min_pixels=5)
                path_layer = pdk.Layer('PathLayer', data=path_layer_data, pickable=True, get_path="path", 
                                       get_width=20, get_color='color', width_min_pixels=2)
                
                MAPBOX_API_KEY = st.secrets.get("mapbox_token")
                tooltip = {"html": "{tooltip}", "style": {"backgroundColor": "rgba(40,40,40,0.8)", "color": "white", "border": "1px solid white", "borderRadius": "5px", "padding": "5px"}}
                
                st.pydeck_chart(pdk.Deck(
                    map_style=pdk.map_styles.MAPBOX_DARK if MAPBOX_API_KEY else 'dark',
                    api_keys={'mapbox': MAPBOX_API_KEY} if MAPBOX_API_KEY else None,
                    initial_view_state=view_state,
                    layers=[path_layer, depot_layer, customer_layer],
                    tooltip=tooltip
                ))
                
                with st.expander('Lihat detail rute per kendaraan'):
                    for route in routes:
                        st.subheader(f"Rute untuk Kendaraan {route['vehicle_id'] + 1} (Estimasi: {route['time']/3600:.2f} jam)")
                        route_display = [ "Depot" if node_idx == 0 else f"Pelanggan-{node_idx}" for node_idx in route['nodes']]
                        st.text(' -> '.join(route_display))
            else:
                st.error('Maaf yaa, aku ga bisa kasih solusi yang optimal dari parameter yang kamu kasih...')
        else:
            st.error('Maaf yaa, model yang kamu pakai ga berhasil di pake nih. Coba kamu cek lagi file modelnya ya...')

with tab3:
    st.header("Tentang Proyek Intelejen Perutean Ini")

    try:
        st.image("https://indonesiainside.id/wp-content/uploads/2019/11/new-york-cabs-future-d7134320.jpg", 
                 caption="Taksi di New York City - Pusat dari Analisis Proyek Ini", use_container_width=True)
    except FileNotFoundError:
        st.warning("Gambar tidak ditemukan. Pastikan file gambar ada di direktori yang sama.")

    st.markdown("""
    Proyek ini adalah sebuah purwarupa (*prototype*) yang dibangun untuk mendemonstrasikan bagaimana teknik *machine learning* dan optimisasi dapat digabungkan untuk memecahkan masalah bisnis yang nyata di dunia logistik.
    """)

    st.subheader("Ringkasan Solusi")
    st.markdown("""
    Solusi ini terdiri dari dua komponen utama:
    1.  **Model Prediksi Durasi**: Sebuah model *machine learning* (XGBoost) yang dilatih pada jutaan data historis perjalanan taksi di NYC. Model ini mampu memprediksi durasi perjalanan dengan mempertimbangkan faktor-faktor seperti jarak, lokasi, jam, dan hari.
    2.  **Model Optimisasi Rute**: Model prediksi durasi kemudian digunakan sebagai "mesin penghitung biaya" untuk algoritma optimisasi **Google OR-Tools**. Algoritma ini secara cerdas mencari kombinasi rute terbaik untuk seluruh armada agar total waktu perjalanan menjadi seminimal mungkin.
    """)

    st.subheader("Teknologi yang Digunakan")
    st.markdown("""
    * **Bahasa Pemrograman**: Python
    * **Analisis Data**: Pandas, NumPy
    * **Visualisasi Data**: Matplotlib, Seaborn, PyDeck
    * **Machine Learning**: Scikit-learn, XGBoost
    * **Optimisasi**: Google OR-Tools
    * **Dashboard Web**: Streamlit
    """)

    with st.expander("Lihat Detail Sumber Data"):
        st.markdown("""
        * **Data Utama**: "[2016 NYC Yellow Cab Trip Record Data](https://www.kaggle.com/c/nyc-taxi-trip-duration)" dari kompetisi Kaggle.
        * **Total Data Latih**: ~1.45 juta baris data perjalanan.
        * **Fitur Kunci**: Koordinat GPS, waktu penjemputan, jumlah penumpang.
        * **Tantangan**: Data mentah mengandung banyak *outlier* (data aneh) seperti perjalanan 0 detik atau perjalanan dengan kecepatan supersonik yang harus dibersihkan sebelum melatih model.
        """)

st.markdown(
    '<div class="footer">© 2025 Muhammad Ketsar Ali Abi Wahid</div>',
    unsafe_allow_html=True
)