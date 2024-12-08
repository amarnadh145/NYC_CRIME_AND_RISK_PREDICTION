import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import requests
from streamlit_option_menu import option_menu
import re
from pygwalker.api.streamlit import StreamlitRenderer
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim 

geolocator = Nominatim(user_agent="crime_prediction_app")
st.set_page_config(
    page_title="AJU_BIGDATA_PROJECT",
    layout="wide", 
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_model(filename):
    return joblib.load(filename)

def load_config(file_path):
    with open(file_path, 'r') as config_file:
        config_str = config_file.read()
    return config_str

@st.cache_data
def load_zone_risk_dict():
    zone_risk_df = pd.read_csv("zone_risk.csv", header=None, names=['zone_id', 'risk'])
    zone_risk_df = zone_risk_df.replace([np.inf, -np.inf], np.nan).dropna()
    return dict(zip(zone_risk_df['zone_id'].astype('Int64'), zone_risk_df['risk']))

# Constants
MIN_LATITUDE, MAX_LATITUDE = 40.49, 62.08
MIN_LONGITUDE, MAX_LONGITUDE = -74.26, -73.68
NUM_DIVISIONS = 5000
LAT_STEP = (MAX_LATITUDE - MIN_LATITUDE) / NUM_DIVISIONS
LON_STEP = (MAX_LONGITUDE - MIN_LONGITUDE) / NUM_DIVISIONS
API_KEY = "AlzaSyr_szPrSBTRKxJQjL__Fy9uJRz9h4vL6mq"

def combine_chunks(chunk_dir, chunk_base_name, combined_output_file):
    chunk_files = [f for f in os.listdir(chunk_dir) if f.startswith(chunk_base_name)]
    chunk_files.sort()
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found starting with {chunk_base_name} in {chunk_dir}")
    with open(combined_output_file, 'wb') as combined_file:
        for chunk_file in chunk_files:
            chunk_path = os.path.join(chunk_dir, chunk_file)
            print(f"Adding {chunk_path} to {combined_output_file}")
            with open(chunk_path, 'rb') as f:
                combined_file.write(f.read())
    print(f"Successfully combined into {combined_output_file}")
def find_zone_id(latitude, longitude):
    if not (MIN_LATITUDE <= latitude <= MAX_LATITUDE) or not (MIN_LONGITUDE <= longitude <= MAX_LONGITUDE):
        return None
    lat_index = int((latitude - MIN_LATITUDE) / LAT_STEP)
    lon_index = int((longitude - MIN_LONGITUDE) / LON_STEP)
    if lat_index == NUM_DIVISIONS:
        lat_index -= 1
    if lon_index == NUM_DIVISIONS:
        lon_index -= 1
    return lat_index * NUM_DIVISIONS + lon_index

def identify_routes_risk_score(all_routes_data):
    route_zones_data = {}
    for route_id, route_data in all_routes_data.items():
        route_coordinates = route_data.get("route_coordinates", [])
        route_zones = [find_zone_id(coord['lat'], coord['long']) for coord in route_coordinates]
        risk_count = sum(zone_risk_dict.get(zone, 0) for zone in route_zones if zone is not None)
        route_zones_data[route_id] = {
            "Coordinate": route_coordinates,
            "distance": route_data.get("total_distance"),
            "time": route_data.get("total_duration"),
            "risk_score": risk_count / len(route_coordinates) if route_coordinates else 0
        }
    return route_zones_data

def get_lat_lng(api_key, address):
    geocode_url = "https://maps.gomaps.pro/maps/api/geocode/json"
    params = {"address": address, "key": api_key}
    response = requests.get(geocode_url, params=params)
    if response.status_code != 200:
        raise Exception(f"Error geocoding address: {response.status_code}, {response.text}")
    geocode_data = response.json()
    if "results" in geocode_data and geocode_data["results"]:
        location = geocode_data["results"][0]["geometry"]["location"]
        return location["lat"], location["lng"]
    else:
        raise Exception("No results found for the provided address")

def get_all_routes_with_coordinates(api_key, origin_address, destination_address):
    origin_lat, origin_lng = get_lat_lng(api_key, origin_address)
    destination_lat, destination_lng = get_lat_lng(api_key, destination_address)

    base_url = "https://maps.gomaps.pro/maps/api/directions/json"
    params = {
        "origin": f"{origin_lat},{origin_lng}",
        "destination": f"{destination_lat},{destination_lng}",
        "mode": "walking",
        "alternatives": "true",
        "key": api_key
    }
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        raise Exception(f"Error fetching directions: {response.status_code}, {response.text}")
    try:
        directions_result = response.json()
    except ValueError:
        raise Exception("Error parsing JSON response from Gomaps Directions API")
    if "routes" not in directions_result:
        raise Exception("No routes found in the Directions API response")
    def clean_html(instruction):
        return re.sub(r'<[^>]*>', '', instruction).strip()
    all_routes_steps = {}
    for route_idx, route in enumerate(directions_result.get("routes", [])):
        steps = route["legs"][0]["steps"]
        total_distance = route["legs"][0]["distance"]["text"]
        total_duration = route["legs"][0]["duration"]["text"]
        route_steps = [f"Total Distance: {total_distance}", f"Total Duration: {total_duration}"]
        route_coordinates = []
        for step in steps:
            instruction = step.get("html_instructions", "No instruction provided.")
            distance = step["distance"]["text"]
            duration = step["duration"]["text"]
            clean_instruction = clean_html(instruction)
            end_location = step["end_location"]
            lat, lng = end_location["lat"], end_location["lng"]
            route_coordinates.append({"lat": lat, "long": lng})
            step_text = f"{clean_instruction} ({distance}, {duration})"
            route_steps.append(step_text)
        all_routes_steps[str(route_idx)] = {
            "steps": route_steps,
            "route_coordinates": route_coordinates,
            "total_distance": total_distance,
            "total_duration": total_duration
        }
    return all_routes_steps

def cleanup_memory():
    import gc
    if 'df' in st.session_state:
        del st.session_state.df
    gc.collect()

# Function to get location name from lat/lng
def get_location_name(lat, lng):
    try:
        location = geolocator.reverse((lat, lng), language="en")
        return location.address if location else "Location not found"
    except Exception as e:
        return f"Error fetching location name: {e}"

st.title("NYC CRIME & ROUTE PREDICTION")
with st.sidebar:
    sel = option_menu(
        menu_title="Navigation",
        options=["HOME","CRIME PREDICTION", "ROUTE PREDICTION"],
        icons=["house", "shield-exclamation", "map"],
        menu_icon="list",
        default_index=0
    )

if sel == "HOME":
    st.header("WELCOME TO OUR CRIME AND ROUTE PREDICTION APP 🚨")
    st.subheader("About Our App")
    st.markdown(
        """WELCOME TO THE **NYC RISK & ROUTE PREDICTION** APP!<br>
        OUR MISSION IS TO ENHANCE PUBLIC SAFETY BY PROVIDING VALUABLE INSIGHTS INTO CRIME PATTERNS AND OPTIMAL TRAVEL ROUTES.""", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image("CRIME_PREDICTION.png", caption="CRIME PREDICTION", width=250)
    with col2:
        st.image("ROUTE_PREDICTION.png", caption="ROUTE PREDICTION", width=250)
    st.markdown("""
        ### HOW IT WORKS
        OUR APP LEVERAGES ADVANCED DATA ANALYTICS AND MACHINE LEARNING TO PROVIDE:

        - **CRIME PREDICTION**: IDENTIFY POTENTIAL CRIME HOTSPOTS AND RISK AREAS USING HISTORICAL DATA AND PREDICTIVE MODELING.
        - **ROUTE PREDICTION**: CHOOSE SAFER TRAVEL PATHS BY CONSIDERING HISTORICAL CRIME DATA.

        EMPOWER YOURSELF WITH DATA-DRIVEN INSIGHTS FOR SAFER AND SMARTER CITY NAVIGATION.
    """)
elif sel == "CRIME PREDICTION":
    st.header("CRIME PREDICTION")
    st.write("PREDICTS PROBABILITY OF A CRIME AND TYPE AT A GIVEN LOCATION")
    address = st.text_input("ENTER ADDRESS", key="address_input")
    map_center = [40.7128, -74.0060]
    m = folium.Map(location=map_center, zoom_start=12)
    folium.Marker(location=map_center, popup="Default NYC Center").add_to(m)
    location = st_folium(m, width=700, height=500)
    if location and location.get("last_clicked"):
        lat, lng = location["last_clicked"]["lat"], location["last_clicked"]["lng"]
        location_name = get_location_name(lat, lng)
        st.write(f"Location: {location_name}")
    elif address:
        try:
            lat, lng = get_lat_lng(API_KEY, address)
            st.write(f"Latitude: {lat}, Longitude: {lng}")
        except Exception as e:
            st.error(f"Error fetching coordinates: {e}")
    if st.button("PREDICT"):
        if not lat or not lng:
            st.error("Please provide a valid location.")
        else:
            try:
                chunk_dir = '.'
                chunk_base_name = "AJU_MODEL_CRIME_RISK_PART"  
                combined_output_file = "AJU_MODEL_CRIME_RISK.pkl"
                combine_chunks(chunk_dir, chunk_base_name, combined_output_file)
                input_df = pd.DataFrame({'Latitude': [lat], 'Longitude': [lng]})
                expected_columns = ['Latitude', 'Longitude']
                input_df = input_df[expected_columns]
                model_crime_risk = load_model(combined_output_file)
                model_crime_type = load_model('AJU_MODEL_CRIME_TYPE.pkl')
                label_encoder_crime = load_model('LABEL_ENCODER_CRIME.pkl')
                crime_risk_prediction = model_crime_risk.predict(input_df)[0]
                st.write(f"Predicted Crime Risk Percentage: {crime_risk_prediction:.2f}%")
                crime_type_probabilities = model_crime_type.predict_proba(input_df)[0]
                predicted_crime_types = label_encoder_crime.inverse_transform(
                    crime_type_probabilities.argsort()[::-1][:3])
                st.write("Top 3 Likely Crime Types:")
                for i, crime_type in enumerate(predicted_crime_types):
                    probability = crime_type_probabilities[crime_type_probabilities.argsort()[::-1][i]] * 100
                    st.write(f"{i + 1}. {crime_type}: {probability:.2f}%")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
elif sel == "ROUTE PREDICTION":
    st.header("ROUTE PREDICTION")
    st.write("PREDICTS RISK SCORES OF THE PATHS BETWEEN A GIVEN SOURCE AND DESTINATION")
    if "source_lat" not in st.session_state:
        st.session_state.source_lat = None
        st.session_state.source_lng = None
    if "destination_lat" not in st.session_state:
        st.session_state.destination_lat = None
        st.session_state.destination_lng = None
    if "map_source" not in st.session_state:
        st.session_state.map_source = folium.Map(location=[40.7128, -74.0060], zoom_start=12)  # Default NYC center
    if "map_destination" not in st.session_state:
        st.session_state.map_destination = folium.Map(location=[40.7128, -74.0060], zoom_start=12)  # Default NYC center
    st.subheader("SOURCE LOCATION")
    source_method = st.radio("SELECT SOURCE METHOD", ("TYPE ADDRESS", "SELECT ON MAP"), key="source_method_unique")
    if source_method == "TYPE ADDRESS":
        source_address = st.text_input("ENTER SOURCE ADDRESS:")
        if source_address:
            try:
                source_lat, source_lng = get_lat_lng(API_KEY, source_address)
                st.session_state.source_lat = source_lat
                st.session_state.source_lng = source_lng
                source_location_name = get_location_name(source_lat, source_lng)
                st.session_state.map_source = folium.Map(location=[source_lat, source_lng], zoom_start=12)
                folium.Marker([source_lat, source_lng], popup=f"Source: {source_location_name}").add_to(st.session_state.map_source)
                st.subheader("SOURCE LOCATION MAP")
                st_folium(st.session_state.map_source, width=700, height=500, key="source_map_unique")
            except Exception as e:
                st.error(f"Error fetching source coordinates: {e}")
    elif source_method == "SELECT ON MAP":
        location = st_folium(st.session_state.map_source, width=700, height=500, key="source_map_unique")
        if location and location.get("last_clicked"):
            st.session_state.source_lat = location["last_clicked"]["lat"]
            st.session_state.source_lng = location["last_clicked"]["lng"]
            source_location_name = get_location_name(st.session_state.source_lat, st.session_state.source_lng)
            st.session_state.map_source = folium.Map(location=[st.session_state.source_lat, st.session_state.source_lng], zoom_start=12)
            folium.Marker([st.session_state.source_lat, st.session_state.source_lng],
                          popup=f"Source: {source_location_name}",
                          draggable=True).add_to(st.session_state.map_source)
            st_folium(st.session_state.map_source, width=700, height=500, key="source_map_updated")
    st.subheader("DESTINATION LOCATION")
    destination_method = st.radio("SELECT DESTINATION METHOD", ("TYPE ADDRESS", "SELECT ON MAP"),key="destination_method_unique")
    if destination_method == "TYPE ADDRESS":
        destination_address = st.text_input("ENTER DESTINATION ADDRESS:")
        if destination_address:
            try:
                destination_lat, destination_lng = get_lat_lng(API_KEY, destination_address)
                st.session_state.destination_lat = destination_lat
                st.session_state.destination_lng = destination_lng
                destination_location_name = get_location_name(destination_lat, destination_lng)
                st.session_state.map_destination = folium.Map(location=[destination_lat, destination_lng], zoom_start=12)
                folium.Marker([destination_lat, destination_lng], popup=f"Destination: {destination_location_name}").add_to(st.session_state.map_destination)
                st.subheader("Destination Location Map")
                st_folium(st.session_state.map_destination, width=700, height=500, key="destination_map_unique")
            except Exception as e:
                st.error(f"Error fetching destination coordinates: {e}")
    elif destination_method == "SELECT ON MAP":
        location = st_folium(st.session_state.map_destination, width=700, height=500, key="destination_map_unique")
        if location and location.get("last_clicked"):
            st.session_state.destination_lat = location["last_clicked"]["lat"]
            st.session_state.destination_lng = location["last_clicked"]["lng"]
            destination_location_name = get_location_name(st.session_state.destination_lat, st.session_state.destination_lng)
            st.session_state.map_destination = folium.Map(location=[st.session_state.destination_lat, st.session_state.destination_lng], zoom_start=12)
            folium.Marker([st.session_state.destination_lat, st.session_state.destination_lng],
                          popup=f"Destination: {destination_location_name}",
                          draggable=True).add_to(st.session_state.map_destination)
            st_folium(st.session_state.map_destination, width=700, height=500, key="destination_map_updated")
    source_location_name = get_location_name(st.session_state.source_lat, st.session_state.source_lng)
    st.write(f"SOURCE LOCATION NAME: {source_location_name}")
    destination_location_name = get_location_name(st.session_state.destination_lat, st.session_state.destination_lng)
    st.write(f"DESTINATION LOCATION NAME: {destination_location_name}")
    if st.session_state.source_lat and st.session_state.source_lng and st.session_state.destination_lat and st.session_state.destination_lng:
        calculate_button = st.button("CALCULATE ROUTES")
        if calculate_button:
            zone_risk_dict = load_zone_risk_dict()
            try:
                all_routes_data = get_all_routes_with_coordinates(API_KEY,
                                                                  f"{st.session_state.source_lat},{st.session_state.source_lng}",
                                                                  f"{st.session_state.destination_lat},{st.session_state.destination_lng}")
                route_risk_scores = identify_routes_risk_score(all_routes_data)
                for route_id, route_info in route_risk_scores.items():
                    st.subheader(f"Route {route_id}")
                    st.write(f"Risk Score: {route_info.get('risk_score', 'Unknown')}")
                    st.write(f"Total Distance: {route_info.get('distance')}")
                    st.write(f"Total Duration: {route_info.get('time')}")
                    for step in all_routes_data[route_id]['steps']:
                        st.write(step)
            except Exception as e:
                st.error(f"An error occurred while calculating routes: {str(e)}")
    else:
        st.error("PLEASE SELECT BOTH SOURCE AND DESTINATION LOCATIONS.")