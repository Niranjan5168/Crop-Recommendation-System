import pandas as pd
import numpy as np
import datetime as dt
import requests
import os
import soundfile as sf
import base64
from io import BytesIO

from flask import Flask, render_template, request, jsonify, send_file
from geopy.geocoders import ArcGIS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

import torch
from transformers import pipeline

# --- Load Models ---
device = "cuda" if torch.cuda.is_available() else "cpu"
stt_pipeline = pipeline("automatic-speech-recognition", model="distil-whisper/distil-large-v2", device=device)
tts_pipeline = pipeline("text-to-speech", model="facebook/mms-tts-kan", device=device)

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
    'rice': 4, 'ragi': 5, 'jowar': 4, 'maize': 4, 'sugarcane': 12, 'cotton': 6, 'groundnut': 4,
    'sunflower': 3, 'pulses': 3, 'coffee': 9, 'arecanut': 7, 'tobacco': 4, 'coconut': 12,
    'banana': 12, 'mango': 10, 'cashew': 8, 'pepper': 6, 'tea': 12, 'turmeric': 8, 'sorghum': 4,
    'paddy': 4, 'jute': 5, 'grapes': 7, 'apple': 8, 'orange': 8, 'chickpea': 4, 'lentil': 4,
    'pigeonpeas': 5, 'mothbeans': 3, 'mungbean': 3, 'blackgram': 3, 'kidneybeans': 4,
    'muskmelon': 3, 'watermelon': 3, 'papaya': 9, 'pomegranate': 9
}
AVERAGE_YIELD_PER_ACRE = {
    'rice': 2400, 'ragi': 1000, 'jowar': 1200, 'maize': 2500, 'sugarcane': 35000, 'cotton': 550,
    'groundnut': 1500, 'sunflower': 800, 'pulses': 400, 'coffee': 300, 'arecanut': 1000,
    'tobacco': 1200, 'coconut': 4800, 'banana': 20000, 'mango': 4000, 'cashew': 500, 'pepper': 200,
    'tea': 1000, 'turmeric': 7000, 'sorghum': 1200, 'paddy': 2400, 'jute': 2000, 'grapes': 9000,
    'apple': 6000, 'orange': 8000, 'chickpea': 800, 'lentil': 600, 'pigeonpeas': 700,
    'mothbeans': 500, 'mungbean': 500, 'blackgram': 600, 'kidneybeans': 900, 'muskmelon': 7000,
    'watermelon': 10000, 'papaya': 15000, 'pomegranate': 6000
}

app = Flask(__name__)

# --- Helper Functions ---
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
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']], data['label'])
    return model

def build_and_train_lstm(X, y):
    model = Sequential([
        Input(shape=(X.shape[1], 1)),
        LSTM(64, activation='relu', return_sequences=True), Dropout(0.2),
        LSTM(32, activation='relu'), Dropout(0.2), Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    return model

def predict_future_price(model, last_sequence, scaler, months_ahead):
    current_sequence = last_sequence.copy()
    for _ in range(months_ahead):
        sequence_for_pred = current_sequence.reshape(1, -1, 1)
        next_pred = model.predict(sequence_for_pred, verbose=0)[0][0]
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred
    final_pred_scaled = np.array([[current_sequence[-1]]])
    return scaler.inverse_transform(final_pred_scaled)[0][0]

def analyze_profitability(crops, district, market_data):
    results = {}
    sowing_month = dt.datetime.now().month
    for crop in crops:
        crop_lower = crop.lower().strip()
        if crop_lower not in GROWTH_PERIODS or crop_lower not in AVERAGE_YIELD_PER_ACRE:
            results[crop] = {'status': 'error', 'message': f"Data not defined for {crop}."}
            continue
        # For demonstration, using simplified revenue. For full functionality, re-enable LSTM.
        # This check avoids the long training time for every request in a demo.
        if crop_lower in AVERAGE_YIELD_PER_ACRE:
            yield_per_acre = AVERAGE_YIELD_PER_ACRE[crop_lower]
            # Using a placeholder price for speed. Replace 25 with LSTM prediction for real use.
            predicted_price = 25 
            total_revenue = predicted_price * yield_per_acre
            results[crop] = {'status': 'success', 'predicted_price': predicted_price, 'yield_per_acre': yield_per_acre, 'total_revenue': total_revenue}
        else:
             results[crop] = {'status': 'error', 'message': f"Yield data not found for {crop}."}
    return results

def run_prediction_logic(address):
    location, full_address = get_location_from_address(address)
    if not location: return {'error': "Location not found."}
    try:
        soil_data_df = pd.read_csv('Copy of soil_data1.csv')
        coordinates_df = pd.read_csv('Copy of district_coordinates.csv')
        market_data_df = pd.read_csv('dataset2.csv')
    except FileNotFoundError as e: return {'error': f"Data file not found: {e.filename}"}
    
    numeric_columns = ['N', 'P', 'K', 'pH', 'Rainfall']
    for col in numeric_columns:
        soil_data_df[col] = pd.to_numeric(soil_data_df[col], errors='coerce')
    soil_data_df.dropna(subset=numeric_columns, inplace=True)
    
    latitude, longitude = location
    district = find_district_knn(latitude, longitude, coordinates_df)
    soil_data_row = soil_data_df[soil_data_df['District'].str.lower() == district.lower()]
    if soil_data_row.empty: return {'error': f"No soil data for {district}."}
    
    weather_data = get_weather_data(district)
    if not weather_data: return {'error': f"No weather data for {district}."}
    
    crop_model = train_crop_recommendation_model()
    data = soil_data_row.iloc[0]
    features = pd.DataFrame([{'N': data['N'], 'P': data['P'], 'K': data['K'], 'temperature': weather_data['Temperature'], 'humidity': weather_data['Humidity'], 'ph': data['pH'], 'rainfall': data['Rainfall']}])
    
    predictions = crop_model.predict_proba(features)[0]
    top_indices = np.argsort(predictions)[-3:][::-1]
    top_3_crops = [crop_model.classes_[i] for i in top_indices]
    
    profit_analysis_results = analyze_profitability(top_3_crops, district, market_data_df)
    most_profitable_crop, highest_revenue = None, -1
    for crop, result in profit_analysis_results.items():
        if result.get('status') == 'success' and result['total_revenue'] > highest_revenue:
            highest_revenue, most_profitable_crop = result['total_revenue'], crop
    if not most_profitable_crop: most_profitable_crop = top_3_crops[0] if top_3_crops else "Not found"
    
    # Return all the data needed for the frontend
    return {
        "status": "success", "district": district, "address": full_address,
        "soil_data": {'N': data['N'], 'P': data['P'], 'K': data['K'], 'pH': data['pH'], 'Rainfall': data['Rainfall']},
        "weather_data": weather_data,
        "top_3_crops": top_3_crops,
        "profit_analysis": profit_analysis_results,
        "most_profitable_crop": most_profitable_crop
    }


# --- FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

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
        return jsonify({"error": "Could not understand audio. Please try again."}), 500

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation_route():
    data = request.get_json()
    address = data.get('text')
    if not address: return jsonify({"error": "No transcribed text provided."}), 400
    
    prediction_result = run_prediction_logic(address)
    if "error" in prediction_result:
        return jsonify(prediction_result), 400
    
    # Prepare spoken response
    crop_english = prediction_result['most_profitable_crop'].lower()
    district_english = prediction_result['district'].lower()
    crop_kannada = KANNADA_TRANSLATIONS.get(crop_english, crop_english)
    district_kannada = KANNADA_TRANSLATIONS.get(district_english, district_english)
    response_text = f"ನಿಮ್ಮ {district_kannada} ಜಿಲ್ಲೆಗೆ, ಅತ್ಯಂತ ಲಾಭದಾಯಕ ಬೆಳೆ {crop_kannada} ಆಗಿದೆ."
    
    try:
        # Generate audio
        output_audio = tts_pipeline(response_text)
        audio_waveform = output_audio["audio"][0]
        samplerate = output_audio["sampling_rate"]
        
        # Convert audio to Base64 string instead of saving to file
        buffer = BytesIO()
        sf.write(buffer, audio_waveform, samplerate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Prepare final JSON response
        final_response = {
            "status": "success",
            "detailed_results": prediction_result,
            "audio_base64": audio_base64
        }
        return jsonify(final_response)

    except Exception as e:
        return jsonify({"error": "Could not generate audio response."}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5001)