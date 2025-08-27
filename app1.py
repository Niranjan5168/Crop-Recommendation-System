import pandas as pd
import numpy as np
import datetime as dt
import requests
import os
import soundfile as sf
import base64
from io import BytesIO
import time
import multiprocessing

from flask import Flask, render_template, request, jsonify
from geopy.geocoders import ArcGIS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import torch
from transformers import pipeline

# --- Load AI/ML Models ---
# Setup device for PyTorch
device = "cuda" if torch.cuda.is_available() else "cpu"

# Speech-to-Text pipeline
print("Loading Speech-to-Text model...")
stt_pipeline = pipeline("automatic-speech-recognition", model="distil-whisper/distil-large-v2", device=device)

# Text-to-Speech pipelines
print("Loading Text-to-Speech models...")
tts_pipeline_kn = pipeline("text-to-speech", model="facebook/mms-tts-kan", device=device)
tts_pipeline_en = pipeline("text-to-speech", model="facebook/mms-tts-eng", device=device)
print("All models loaded.")

# --- Dictionaries and Constants ---
KANNADA_TRANSLATIONS = {
    'rice': 'ಭತ್ತ', 'ragi': 'ರಾಗಿ', 'jowar': 'ಜೋಳ', 'maize': 'ಮೆಕ್ಕೆಜೋಳ', 'sugarcane': 'ಕಬ್ಬು',
    'cotton': 'ಹತ್ತಿ', 'groundnut': 'ಕಡಲೆಕಾಯಿ', 'sunflower': 'ಸೂರ್ಯಕಾಂತಿ', 'pulses': 'ದ್ವಿದಳ ಧಾನ್ಯಗಳು',
    'coffee': 'ಕಾಫಿ', 'arecanut': 'ಅಡಿಕೆ', 'tobacco': 'ತಂಬಾಕು', 'coconut': 'ತೆಂಗಿನಕಾಯಿ',
    'banana': 'ಬಾಳೆಹಣ್ಣು', 'mango': 'ಮಾವಿನಹಣ್ಣು', 'cashew': 'ಗೋಡಂಬಿ', 'pepper': 'ಮೆಣಸು',
    'tea': 'ಚಹಾ', 'turmeric': 'ಅರಿಶಿನ', 'sorghum': 'ಜೋಳ', 'paddy': 'ಭತ್ತ',
    'grapes': 'ದ್ರಾಕ್ಷಿ', 'apple': 'ಸೇಬು', 'orange': 'ಕಿತ್ತಳೆ', 'chickpea': 'ಕಡಲೆ',
    'pigeonpeas': 'ತೊಗರಿ', 'mungbean': 'ಹೆಸರು ಕಾಳು', 'blackgram': 'ಉದ್ದಿನ ಬೇಳೆ',
    'kidneybeans': 'ರಾಜಮಾ', 'watermelon': 'ಕಲ್ಲಂಗಡಿ', 'muskmelon': 'ಕರಬೂಜ', 'papaya': 'ಪಪ್ಪಾಯಿ',
    'pomegranate': 'ದಾಳಿಂಬೆ',
    'mysuru': 'ಮೈಸೂರು', 'bengaluru urban': 'ಬೆಂಗಳೂರು ನಗರ', 'hassan': 'ಹಾಸನ', 'mandya': 'ಮಂಡ್ಯ',
    'chikkamagaluru': 'ಚಿಕ್ಕಮಗಳೂರು', 'shivamogga': 'ಶಿವಮೊಗ್ಗ', 'belagavi': 'ಬೆಳಗಾವಿ',
    'dharwad': 'ಧಾರವಾಡ', 'kalaburagi': 'ಕಲಬುರಗಿ', 'dakshina kannada': 'ದಕ್ಷಿಣ ಕನ್ನಡ'
}
BASE_URL = "http://api.openweathermap.org/data/2.5/weather?"
API_KEY = '97fcd3c0d35e200184c08eac1a53f987'
GROWTH_PERIODS = {
    # Using capitalized keys to match 'dataset2.csv'
    'Rice': 4, 'Ragi': 5, 'Jowar': 4, 'Maize': 4, 'Sugarcane': 12, 'Cotton': 6, 'Groundnut': 4,
    'Sunflower': 3, 'Pulses': 3, 'Coffee': 9, 'Arecanut': 7, 'Tobacco': 4, 'Coconut': 12,
    'Banana': 12, 'Mango': 10, 'Cashew': 8, 'Pepper': 6, 'Tea': 12, 'Turmeric': 8, 'Sorghum': 4,
    'Paddy': 4, 'Jute': 5, 'Grapes': 7, 'Apple': 8, 'Orange': 8, 'Chickpea': 4, 'Lentil': 4,
    'Pigeonpeas': 5, 'Mothbeans': 3, 'Mungbean': 3, 'Blackgram': 3, 'Kidneybeans': 4,
    'Muskmelon': 3, 'Watermelon': 3, 'Papaya': 9, 'Pomegranate': 9
}

app = Flask(__name__)

# --- Helper Functions (Original) ---
def get_weather_data(city):
    url = BASE_URL + "appid=" + API_KEY + "&q=" + city
    try:
        response = requests.get(url).json()
        if 'main' not in response: return None
        return {'Temperature': response['main']['temp'] - 273.15, 'Humidity': response['main']['humidity']}
    except Exception: return None

def get_location_from_address(address):
    geolocator = ArcGIS()
    try:
        location = geolocator.geocode(address, timeout=10)
        return (location.latitude, location.longitude), location.address if location else (None, None)
    except Exception: return None, None

def find_district_knn(lat, lon, coordinates_df):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(coordinates_df[['Latitude', 'Longitude']].values, coordinates_df['District'])
    return knn.predict(np.array([[lat, lon]]))[0]

def train_crop_recommendation_model():
    data = pd.read_csv('Copy of CROPP.csv')
    # Map labels to capitalized versions to match GROWTH_PERIODS keys
    data['label'] = data['label'].str.capitalize()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']], data['label'])
    return model

# --- LSTM Helper Functions (Integrated) ---
def preprocess_data_for_lstm(market_data, crop, district, look_back=6):
    crop_data = market_data[
        (market_data['Crop Name'].str.lower() == crop.lower()) & 
        (market_data['District'].str.lower() == district.lower())
    ].copy()
    
    if len(crop_data) < look_back + 1:
        raise ValueError(f"Insufficient data for '{crop}' in '{district}'. Need at least {look_back + 1} months.")
    
    crop_data['Date'] = pd.to_datetime(crop_data['Date'], format='%Y-%m')
    crop_data = crop_data.sort_values(by='Date')
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices = crop_data['Market Price (per kg)'].values.reshape(-1, 1)
    normalized_prices = scaler.fit_transform(prices)
    
    X, y = [], []
    for i in range(len(normalized_prices) - look_back):
        X.append(normalized_prices[i:(i + look_back)])
        y.append(normalized_prices[i + look_back])
    
    return np.array(X), np.array(y), scaler

def build_and_train_lstm(X, y):
    model = Sequential([
        Input(shape=(X.shape[1], 1)),
        LSTM(64, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=50, batch_size=16, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    return model

def predict_future_price(model, last_sequence, scaler, months_ahead):
    current_sequence = last_sequence.copy().flatten()
    for _ in range(months_ahead):
        sequence_for_pred = current_sequence.reshape(1, -1, 1)
        next_pred = model.predict(sequence_for_pred, verbose=0)[0][0]
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred
    return scaler.inverse_transform([[current_sequence[-1]]])[0][0]

def analyze_profitability_with_lstm(market_data, crops, district):
    profits = {}
    print("\nAnalyzing profitability with LSTM...")
    for crop in crops:
        # Ensure crop name capitalization matches GROWTH_PERIODS keys
        crop_capitalized = crop.capitalize()
        if crop_capitalized not in GROWTH_PERIODS:
            print(f"Warning: Growth period not defined for crop: {crop_capitalized}")
            continue
        try:
            print(f"  - Processing {crop_capitalized}...")
            X, y, scaler = preprocess_data_for_lstm(market_data, crop_capitalized, district)
            if len(X) == 0: continue
            
            model = build_and_train_lstm(X, y)
            
            growth_duration = GROWTH_PERIODS[crop_capitalized]
            predicted_price = predict_future_price(model, X[-1], scaler, growth_duration)
            profits[crop_capitalized] = round(predicted_price, 2)
            print(f"    Predicted harvest price for {crop_capitalized}: ₹{predicted_price:.2f}")
            
        except Exception as e:
            print(f"    Warning: Could not process {crop_capitalized}: {e}")
            profits[crop_capitalized] = 0
    return profits

# --- Main Logic Function ---
# --- Main Logic Function ---
def run_prediction_logic(address):
    location, full_address = get_location_from_address(address)
    if not location: return {'error': "Location not found."}
    try:
        soil_data_df = pd.read_csv('Copy of soil_data1.csv')
        coordinates_df = pd.read_csv('Copy of district_coordinates.csv')
        market_data_df = pd.read_csv('dataset2.csv')
    except FileNotFoundError as e: return {'error': f"Data file not found: {e.filename}"}
    
    latitude, longitude = location
    district = find_district_knn(latitude, longitude, coordinates_df)
    soil_data_row = soil_data_df[soil_data_df['District'].str.lower() == district.lower()]
    if soil_data_row.empty: return {'error': f"No soil data for {district}."}
    
    weather_data = get_weather_data(district)
    if not weather_data: return {'error': f"No weather data for {district}."}
    
    # --- Part 1: Crop Suitability Prediction (Random Forest) ---
    print(f"Finding suitable crops for {district}...")
    crop_model = train_crop_recommendation_model()
    data = soil_data_row.iloc[0]

    # MODIFIED: Explicitly convert data to numeric types here
    soil_info = {
        'N': int(data['N']),
        'P': int(data['P']),
        'K': int(data['K']),
        'pH': float(data['pH']),
        'Rainfall': float(data['Rainfall'])
    }
    
    # Use the converted values from soil_info for the features
    features = pd.DataFrame([{
        'N': soil_info['N'], 
        'P': soil_info['P'], 
        'K': soil_info['K'], 
        'temperature': weather_data['Temperature'], 
        'humidity': weather_data['Humidity'], 
        'ph': soil_info['pH'], 
        'rainfall': soil_info['Rainfall']
    }])
    
    predictions = crop_model.predict_proba(features)[0]
    top_indices = np.argsort(predictions)[-3:][::-1]
    top_3_crops = [crop_model.classes_[i] for i in top_indices]
    print(f"Top 3 suitable crops: {top_3_crops}")

    # --- Part 2: Profitability Analysis (LSTM) ---
    predicted_prices = analyze_profitability_with_lstm(market_data_df, top_3_crops, district)
    
    if not predicted_prices:
        return {'error': "Could not predict prices for any suitable crop."}
        
    most_profitable_crop = max(predicted_prices, key=predicted_prices.get)
    highest_price = predicted_prices[most_profitable_crop]
    
    return {
        "status": "success", "district": district, "address": full_address,
        "soil_data": soil_info, "weather_data": weather_data,
        "top_3_crops": top_3_crops,
        "predicted_prices": predicted_prices,
        "most_profitable_crop": most_profitable_crop,
        "highest_predicted_price": highest_price
    }


# --- FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/transcribe', methods=['POST'])
def transcribe_route():
    if 'audio_data' not in request.files: return jsonify({"error": "No audio file found."}), 400
    audio_file = request.files['audio_data']
    filepath = os.path.join("temp_audio.wav")
    audio_file.save(filepath)
    try:
        result = stt_pipeline(filepath, generate_kwargs={"language": "kannada"})
        transcribed_text = result["text"].strip()
        if os.path.exists(filepath): os.remove(filepath)
        return jsonify({'transcribed_text': transcribed_text})
    except Exception as e:
        if os.path.exists(filepath): os.remove(filepath)
        return jsonify({"error": f"Could not understand audio: {e}"}), 500

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation_route():
    data = request.get_json()
    address = data.get('text')
    lang = data.get('lang', 'en')
    
    if not address: return jsonify({"error": "No transcribed text provided."}), 400
    
    prediction_result = run_prediction_logic(address)
    if "error" in prediction_result:
        return jsonify(prediction_result), 400
    
    # --- Prepare spoken response based on language ---
    crop_english = prediction_result['most_profitable_crop'].lower()
    district_english = prediction_result['district'].lower()
    price = prediction_result['highest_predicted_price']

    if lang == 'kn':
        crop_kannada = KANNADA_TRANSLATIONS.get(crop_english, crop_english)
        district_kannada = KANNADA_TRANSLATIONS.get(district_english, district_english)
        response_text = f"ನಿಮ್ಮ {district_kannada} ಜಿಲ್ಲೆಗೆ, ಅತ್ಯಂತ ಲಾಭದಾಯಕ ಬೆಳೆ {crop_kannada} ಆಗಿದೆ, ಅದರ ಕೊಯ್ಲು ಬೆಲೆ ಪ್ರತಿ ಕೆಜಿಗೆ {price:.2f} ರೂಪಾಯಿ ಎಂದು ಅಂದಾಜಿಸಲಾಗಿದೆ."
        tts_pipeline_to_use = tts_pipeline_kn
    else: # Default to English
        response_text = f"For your district, {district_english.capitalize()}, the most profitable crop is {crop_english.capitalize()}, with a predicted harvest price of {price:.2f} rupees per kg."
        tts_pipeline_to_use = tts_pipeline_en
        
    try:
        output_audio = tts_pipeline_to_use(response_text)
        audio_waveform = output_audio["audio"][0]
        samplerate = output_audio["sampling_rate"]
        
        buffer = BytesIO()
        sf.write(buffer, audio_waveform, samplerate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        final_response = {
            "status": "success",
            "detailed_results": prediction_result,
            "audio_base64": audio_base64
        }
        return jsonify(final_response)

    except Exception as e:
        return jsonify({"error": f"Could not generate audio response: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)