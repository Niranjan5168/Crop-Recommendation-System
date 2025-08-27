# Agri-Voice Assist: AI-Powered Crop Profitability Forecaster

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/) [![Flask](https://img.shields.io/badge/Flask-2.x-black.svg)](https://flask.palletsprojects.com/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://www.tensorflow.org/)

Agri-Voice Assist is an intelligent web application designed to provide hyper-local, profitable crop recommendations to farmers in Karnataka, India. Using a voice-first interface that supports both **Kannada and English**, the application analyzes local soil, weather, and historical market data to not only suggest suitable crops but also forecast their future profitability.


---

## ✨ Features

-   **🗣️ Multilingual Voice Interface**: Interact with the application by speaking your location in either Kannada or English.
-   **📍 Hyper-Local Analysis**: Get recommendations based on your specific location's soil composition, rainfall, and real-time weather.
-   **🤖 AI-Powered Crop Suitability**: A Random Forest model recommends the top 3 crops best suited for your land.
-   **📈 Profitability Forecasting**: A sophisticated LSTM neural network predicts the market price of recommended crops at harvest time.
-   **🔊 Audio Feedback**: Receive the final recommendation as a clear, spoken response in your chosen language.
-   **📊 Detailed Web Reports**: View comprehensive results, including soil data, weather conditions, and price predictions, on a clean web interface.

---

## ⚙️ How It Works

The application follows a multi-step AI pipeline to generate its recommendations:

1.  **Speech-to-Text**: User's spoken location is transcribed using the `distil-whisper/distil-large-v2` model.
2.  **Geolocation**: The transcribed address is converted to latitude and longitude using `Geopy`.
3.  **District Identification**: A k-Nearest Neighbors (k-NN) model maps the coordinates to a specific district in Karnataka.
4.  **Data Aggregation**: The system fetches soil data (`N`, `P`, `K`, `pH`), rainfall data from local datasets, and live weather (`temperature`, `humidity`) from the OpenWeatherMap API.
5.  **Crop Recommendation**: A trained `RandomForestClassifier` predicts the top 3 most suitable crops.
6.  **Price Prediction**: An `LSTM` model, trained on historical market prices, forecasts the harvest-time price for each of the top 3 crops.
7.  **Translation & TTS**: The final recommendation is translated to Kannada if needed, and then converted to speech using Facebook's `MMS-TTS` models.

---

## 🛠️ Technology Stack

-   **Backend**: Flask
-   **Machine Learning**: Scikit-learn (RandomForestClassifier, KNeighborsClassifier)
-   **Deep Learning**: TensorFlow / Keras (LSTM for time-series forecasting)
-   **AI Pipelines**: PyTorch, Hugging Face Transformers
    -   **Speech-to-Text**: `distil-whisper/distil-large-v2`
    -   **Text-to-Speech**: `facebook/mms-tts-kan` (Kannada), `facebook/mms-tts-eng` (English)
    -   **Translation**: `Helsinki-NLP/opus-mt-en-kn` (or similar English-to-Kannada model)
-   **Data Handling**: Pandas, NumPy
-   **APIs**: OpenWeatherMap

---

## 🚀 Setup and Installation

### Prerequisites

-   Python 3.9+
-   PyTorch
-   TensorFlow
-   An OpenWeatherMap API Key

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/agri-voice-assist.git](https://github.com/your-username/agri-voice-assist.git)
cd agri-voice-assist
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all required packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Set Up Local Translation Model

The application requires a local model for English-to-Kannada translation.

1.  Create a directory for the model, e.g., `models/english-kannada-mt`.
2.  Download a suitable model from the Hugging Face Hub. We recommend `Helsinki-NLP/opus-mt-en-kn`. You can use the following Python script to download it:

    ```python
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_name = "Helsinki-NLP/opus-mt-en-kn"
    save_directory = "models/english-kannada-mt"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)
    print(f"Model saved to {save_directory}")
    ```

3.  Update the model path in `app.py` to point to your new directory:

    ```python
    # In app.py
    translation_tokenizer = AutoTokenizer.from_pretrained("models/english-kannada-mt")
    translation_model = AutoModelForSeq2SeqLM.from_pretrained("models/english-kannada-mt").to(device)
    ```

### 5. Add API Key

Create a `.env` file in the root directory and add your OpenWeatherMap API key:

```
OPENWEATHERMAP_API_KEY="97fcd3c0d35e200184c08eac1a53f987"
```
*(Note: You will need to update the `get_weather_data` function to load this key from the environment variable instead of having it hardcoded.)*

### 6. Place Datasets

Ensure the following CSV files are in the root directory of the project:
- `Copy of CROPP.csv`
- `Copy of soil_data1.csv`
- `Copy of district_coordinates.csv`
- `dataset2.csv`

### 7. Run the Application

```bash
app.py
```

The application will be available at `http://127.0.0.1:5001`.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
