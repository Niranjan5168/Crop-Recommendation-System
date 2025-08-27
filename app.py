import pandas as pd
import numpy as np
import datetime as dt
import requests
import os
import soundfile as sf
import base64
from io import BytesIO
import time

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
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Speech-to-Text model...")
stt_pipeline = pipeline("automatic-speech-recognition", model="distil-whisper/distil-large-v2", device=device)

print("Loading Text-to-Speech models...")
tts_pipeline_kn = pipeline("text-to-speech", model="facebook/mms-tts-kan", device=device)
tts_pipeline_en = pipeline("text-to-speech", model="facebook/mms-tts-eng", device=device)

print("Loading English-Kannada translation model from local directory...")
translation_tokenizer = AutoTokenizer.from_pretrained("/Users/510msqkm/Desktop/MMXNLP/english-kannada-mt")
translation_model = AutoModelForSeq2SeqLM.from_pretrained("/Users/510msqkm/Desktop/MMXNLP/english-kannada-mt").to(device)

def translate_to_kannada(text):
    inputs = translation_tokenizer(text, return_tensors="pt", padding=True).to(device)
    outputs = translation_model.generate(**inputs)
    return translation_tokenizer.decode(outputs[0], skip_special_tokens=True)

GROWTH_PERIODS = {
    'Rice': 4, 'Ragi': 5, 'Jowar': 4, 'Maize': 4, 'Sugarcane': 12, 'Cotton': 6, 'Groundnut': 4,
    'Sunflower': 3, 'Pulses': 3, 'Coffee': 9, 'Arecanut': 7, 'Tobacco': 4, 'Coconut': 12,
    'Banana': 12, 'Mango': 10, 'Cashew': 8, 'Pepper': 6, 'Tea': 12, 'Turmeric': 8, 'Sorghum': 4,
    'Paddy': 4, 'Jute': 5, 'Grapes': 7, 'Apple': 8, 'Orange': 8, 'Chickpea': 4, 'Lentil': 4,
    'Pigeonpeas': 5, 'Mothbeans': 3, 'Mungbean': 3, 'Blackgram': 3, 'Kidneybeans': 4,
    'Muskmelon': 3, 'Watermelon': 3, 'Papaya': 9, 'Pomegranate': 9
}

app = Flask(__name__)

# --- Helper functions ---
def get_weather_data(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?appid=97fcd3c0d35e200184c08eac1a53f987&q={city}"
    try:
        response = requests.get(url).json()
        if 'main' not in response: return None
        return {'Temperature': response['main']['temp'] - 273.15, 'Humidity': response['main']['humidity']}
    except: return None

def get_location_from_address(address):
    geolocator = ArcGIS()
    try:
        location = geolocator.geocode(address, timeout=10)
        return (location.latitude, location.longitude), location.address if location else (None, None)
    except: return None, None

def find_district_knn(lat, lon, coordinates_df):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(coordinates_df[['Latitude', 'Longitude']].values, coordinates_df['District'])
    return knn.predict(np.array([[lat, lon]]))[0]

def train_crop_recommendation_model():
    data = pd.read_csv('Copy of CROPP.csv')
    data['label'] = data['label'].str.capitalize()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']], data['label'])
    return model

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
    for crop in crops:
        crop_cap = crop.capitalize()
        if crop_cap not in GROWTH_PERIODS: continue
        try:
            X, y, scaler = preprocess_data_for_lstm(market_data, crop_cap, district)
            if len(X) == 0: continue
            model = build_and_train_lstm(X, y)
            price = predict_future_price(model, X[-1], scaler, GROWTH_PERIODS[crop_cap])
            profits[crop_cap] = round(price, 2)
        except: profits[crop_cap] = 0
    return profits

def run_prediction_logic(address):
    location, full_address = get_location_from_address(address)
    if not location: return {'error': "Location not found."}
    
    try:
        soil_data_df = pd.read_csv('Copy of soil_data1.csv')
        coordinates_df = pd.read_csv('Copy of district_coordinates.csv')
        market_data_df = pd.read_csv('dataset2.csv')
    except FileNotFoundError as e: return {'error': f"Data file not found: {e.filename}"}
    
    lat, lon = location
    district = find_district_knn(lat, lon, coordinates_df)
    row = soil_data_df[soil_data_df['District'].str.lower() == district.lower()]
    if row.empty: return {'error': f"No soil data for {district}."}
    
    weather = get_weather_data(district)
    if not weather: return {'error': f"No weather data for {district}."}
    
    crop_model = train_crop_recommendation_model()
    data = row.iloc[0]
    features = pd.DataFrame([{
        'N': int(data['N']), 'P': int(data['P']), 'K': int(data['K']),
        'temperature': weather['Temperature'], 'humidity': weather['Humidity'],
        'ph': float(data['pH']), 'rainfall': float(data['Rainfall'])
    }])
    
    pred = crop_model.predict_proba(features)[0]
    top = np.argsort(pred)[-3:][::-1]
    top_crops = [crop_model.classes_[i] for i in top]
    
    prices = analyze_profitability_with_lstm(market_data_df, top_crops, district)
    if not prices: return {'error': "Could not predict prices for any crop."}
    
    best_crop = max(prices, key=prices.get)
    
    return {
        "status": "success", "district": district, "address": full_address,
        "soil_data": features.iloc[0].to_dict(), "weather_data": weather,
        "top_3_crops": top_crops, "predicted_prices": prices,
        "most_profitable_crop": best_crop,
        "highest_predicted_price": prices[best_crop]
    }

@app.route('/')
def index(): return render_template('index.html')

@app.route('/results')
def results(): return render_template('results.html')

@app.route('/about')
def about(): return render_template('about.html')

@app.route('/transcribe', methods=['POST'])
def transcribe_route():
    if 'audio_data' not in request.files: return jsonify({"error": "No audio file found."}), 400
    audio_file = request.files['audio_data']
    filepath = "temp_audio.wav"
    audio_file.save(filepath)
    try:
        result = stt_pipeline(filepath, generate_kwargs={"language": "kannada"})
        if os.path.exists(filepath): os.remove(filepath)
        return jsonify({'transcribed_text': result["text"].strip()})
    except Exception as e:
        if os.path.exists(filepath): os.remove(filepath)
        return jsonify({"error": f"Could not understand audio: {e}"}), 500

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation_route():
    data = request.get_json()
    address = data.get('text')
    lang = data.get('lang', 'en')
    
    if not address: return jsonify({"error": "No transcribed text provided."}), 400
    
    result = run_prediction_logic(address)
    if "error" in result: return jsonify(result), 400
    
    crop_en = result['most_profitable_crop']
    district_en = result['district']
    price = result['highest_predicted_price']
    
    if lang == 'kn':
        crop_kn = translate_to_kannada(crop_en)
        district_kn = translate_to_kannada(district_en)
        response_text = f"ನಿಮ್ಮ {district_kn} ಜಿಲ್ಲೆಗೆ, ಅತ್ಯಂತ ಲಾಭದಾಯಕ ಬೆಳೆ {crop_kn} ಆಗಿದೆ, ಅದರ ಕೊಯ್ಲು ಬೆಲೆ ಪ್ರತಿ ಕೆಜಿಗೆ ₹{price:.2f} ಎಂದು ಅಂದಾಜಿಸಲಾಗಿದೆ."
        tts = tts_pipeline_kn
    else:
        response_text = f"For your district, {district_en}, the most profitable crop is {crop_en}, with a predicted harvest price of ₹{price:.2f} per kg."
        tts = tts_pipeline_en
    
    try:
        output_audio = tts(response_text)
        waveform = output_audio["audio"][0]
        sr = output_audio["sampling_rate"]
        buffer = BytesIO()
        sf.write(buffer, waveform, sr, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return jsonify({
            "status": "success", "detailed_results": result, "audio_base64": audio_base64
        })
    except Exception as e:
        return jsonify({"error": f"Could not generate audio response: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)