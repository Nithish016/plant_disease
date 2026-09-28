# Importing essential libraries and modules

import os
from flask import Flask, render_template, request
try:
    from flask import Markup
except ImportError:
    from markupsafe import Markup
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu'), weights_only=False))
disease_model.eval()


# Loading crop recommendation model

crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'Kisan AI - Home'
    return render_template('index.html', title=title)

# render login page using Clerk
@app.route('/login')
def login():
    title = 'Kisan AI - Login'
    clerk_key = getattr(config, 'clerk_publishable_key', os.environ.get('CLERK_PUBLISHABLE_KEY', 'pk_test_Y2xlcmsuZXhhbXBsZS5jb20k'))
    return render_template('login.html', title=title, clerk_publishable_key=clerk_key)

# render crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Kisan AI - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Kisan AI - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

# render disease prediction input page




# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Kisan AI - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, title=title)

        else:

            return render_template('try_again.html', title=title)

# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Kisan AI - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Kisan AI - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)


# ===============================================================================================
# -------------------- COMMODITY MARKET PRICES & BUDGET PREDICTOR ------------------------------
# ===============================================================================================

COMMODITY_PRICES = [
    {
        'id': 'wheat',
        'name_en': 'Wheat (Lokwan / Sharbati)',
        'name_hi': 'गेहूं',
        'mandi': 'Khanna / Indore Mandi',
        'price': 2480,
        'msp': 2275,
        'change': '+1.8%',
        'change_val': 45,
        'is_positive': True,
        'high': 2520,
        'low': 2420,
        'arrival_quintals': 12500,
        'trend': 'Bullish'
    },
    {
        'id': 'paddy',
        'name_en': 'Paddy / Rice (Common & Basmati)',
        'name_hi': 'धान / चावल',
        'mandi': 'Karnal / Warangal Mandi',
        'price': 2360,
        'msp': 2183,
        'change': '+0.9%',
        'change_val': 20,
        'is_positive': True,
        'high': 2410,
        'low': 2320,
        'arrival_quintals': 18900,
        'trend': 'Bullish'
    },
    {
        'id': 'cotton',
        'name_en': 'Cotton (Medium / Long Staple)',
        'name_hi': 'कपास',
        'mandi': 'Rajkot / Adilabad Mandi',
        'price': 7150,
        'msp': 6620,
        'change': '-0.7%',
        'change_val': -50,
        'is_positive': False,
        'high': 7280,
        'low': 7080,
        'arrival_quintals': 8400,
        'trend': 'Stable'
    },
    {
        'id': 'maize',
        'name_en': 'Maize (Corn)',
        'name_hi': 'मक्का',
        'mandi': 'Gulabbagh / Davanagere Mandi',
        'price': 2120,
        'msp': 2090,
        'change': '+2.4%',
        'change_val': 50,
        'is_positive': True,
        'high': 2160,
        'low': 2060,
        'arrival_quintals': 9200,
        'trend': 'Bullish'
    },
    {
        'id': 'soybean',
        'name_en': 'Soybean (Yellow)',
        'name_hi': 'सोयाबीन',
        'mandi': 'Ujjain / Latur Mandi',
        'price': 4720,
        'msp': 4600,
        'change': '-1.2%',
        'change_val': -60,
        'is_positive': False,
        'high': 4830,
        'low': 4680,
        'arrival_quintals': 14200,
        'trend': 'Bearish'
    },
    {
        'id': 'mustard',
        'name_en': 'Mustard Seed (Sarson)',
        'name_hi': 'सरसों',
        'mandi': 'Jaipur / Bharatpur Mandi',
        'price': 5480,
        'msp': 5650,
        'change': '+1.1%',
        'change_val': 60,
        'is_positive': True,
        'high': 5540,
        'low': 5400,
        'arrival_quintals': 11300,
        'trend': 'Bullish'
    },
    {
        'id': 'tomato',
        'name_en': 'Tomato (Hybrid Red)',
        'name_hi': 'टमाटर',
        'mandi': 'Kolar / Madanapalle Mandi',
        'price': 2850,
        'msp': 1800,
        'change': '+6.2%',
        'change_val': 165,
        'is_positive': True,
        'high': 3100,
        'low': 2600,
        'arrival_quintals': 6800,
        'trend': 'High Demand'
    },
    {
        'id': 'potato',
        'name_en': 'Potato (Jyoti / Pukhraj)',
        'name_hi': 'आलू',
        'mandi': 'Agra / Farrukhabad Mandi',
        'price': 1840,
        'msp': 1400,
        'change': '+0.5%',
        'change_val': 10,
        'is_positive': True,
        'high': 1900,
        'low': 1780,
        'arrival_quintals': 22000,
        'trend': 'Stable'
    },
    {
        'id': 'chana',
        'name_en': 'Gram / Chickpea (Desi Chana)',
        'name_hi': 'चना',
        'mandi': 'Bikaner / Akola Mandi',
        'price': 5890,
        'msp': 5440,
        'change': '+1.5%',
        'change_val': 90,
        'is_positive': True,
        'high': 5960,
        'low': 5790,
        'arrival_quintals': 7500,
        'trend': 'Bullish'
    },
    {
        'id': 'sugarcane',
        'name_en': 'Sugarcane (FRP Mill Gate)',
        'name_hi': 'गन्ना',
        'mandi': 'Kolhapur / Muzaffarnagar',
        'price': 340,
        'msp': 315,
        'change': '0.0%',
        'change_val': 0,
        'is_positive': True,
        'high': 350,
        'low': 335,
        'arrival_quintals': 45000,
        'trend': 'Stable'
    }
]

BUDGET_BENCHMARKS = {
    'wheat': {'name': 'Wheat (गेहूं)', 'yield_per_acre': 22, 'mandi_price': 2480, 'seed_cost': 2200, 'fert_cost': 3400, 'irrigation_cost': 2100, 'pest_cost': 1400, 'machinery_cost': 3800, 'labor_cost': 3200},
    'paddy': {'name': 'Paddy / Rice (धान)', 'yield_per_acre': 26, 'mandi_price': 2360, 'seed_cost': 1800, 'fert_cost': 3900, 'irrigation_cost': 3800, 'pest_cost': 1800, 'machinery_cost': 4200, 'labor_cost': 4500},
    'cotton': {'name': 'Cotton (कपास)', 'yield_per_acre': 12, 'mandi_price': 7150, 'seed_cost': 2800, 'fert_cost': 4200, 'irrigation_cost': 2400, 'pest_cost': 3200, 'machinery_cost': 3500, 'labor_cost': 5500},
    'maize': {'name': 'Maize / Corn (मक्का)', 'yield_per_acre': 28, 'mandi_price': 2120, 'seed_cost': 2400, 'fert_cost': 3600, 'irrigation_cost': 2000, 'pest_cost': 1200, 'machinery_cost': 3600, 'labor_cost': 2800},
    'soybean': {'name': 'Soybean (सोयाबीन)', 'yield_per_acre': 13, 'mandi_price': 4720, 'seed_cost': 3100, 'fert_cost': 2800, 'irrigation_cost': 1200, 'pest_cost': 1600, 'machinery_cost': 3200, 'labor_cost': 2600},
    'mustard': {'name': 'Mustard (सरसों)', 'yield_per_acre': 9, 'mandi_price': 5480, 'seed_cost': 1200, 'fert_cost': 2500, 'irrigation_cost': 1400, 'pest_cost': 1100, 'machinery_cost': 2900, 'labor_cost': 2200},
    'tomato': {'name': 'Tomato (टमाटर)', 'yield_per_acre': 115, 'mandi_price': 2850, 'seed_cost': 5500, 'fert_cost': 6800, 'irrigation_cost': 3500, 'pest_cost': 4500, 'machinery_cost': 4000, 'labor_cost': 9500},
    'potato': {'name': 'Potato (आलू)', 'yield_per_acre': 95, 'mandi_price': 1840, 'seed_cost': 8500, 'fert_cost': 5800, 'irrigation_cost': 2800, 'pest_cost': 3200, 'machinery_cost': 4500, 'labor_cost': 6200},
    'chana': {'name': 'Chickpea / Chana (चना)', 'yield_per_acre': 11, 'mandi_price': 5890, 'seed_cost': 2600, 'fert_cost': 2200, 'irrigation_cost': 1300, 'pest_cost': 1500, 'machinery_cost': 3000, 'labor_cost': 2400}
}


@app.route('/market')
def market_prices():
    title = 'Kisan AI - Mandi & Stock Market Prices'
    return render_template('market.html', title=title, prices=COMMODITY_PRICES)


@app.route('/budget-predictor', methods=['GET', 'POST'])
def budget_predictor():
    title = 'Kisan AI - Farm Budget & Profit Predictor'
    crop_key = request.form.get('crop', 'wheat').lower() if request.method == 'POST' else 'wheat'
    acres = float(request.form.get('acres', 2.0)) if request.method == 'POST' else 2.0
    irrigation_type = request.form.get('irrigation_type', 'drip') if request.method == 'POST' else 'drip'
    labor_mode = request.form.get('labor_mode', 'mechanized') if request.method == 'POST' else 'mechanized'
    tillage_mode = request.form.get('tillage_mode', 'tractor') if request.method == 'POST' else 'tractor'

    bench = BUDGET_BENCHMARKS.get(crop_key, BUDGET_BENCHMARKS['wheat'])

    irrigation_multiplier = 0.75 if irrigation_type == 'drip' else (0.85 if irrigation_type == 'sprinkler' else 1.15)
    labor_multiplier = 0.70 if labor_mode == 'mechanized' else (0.50 if labor_mode == 'family' else 1.25)
    machinery_multiplier = 1.20 if tillage_mode == 'heavy_tractor' else (0.80 if tillage_mode == 'zero_till' else 1.0)

    seed_cost_acre = bench['seed_cost']
    fert_cost_acre = bench['fert_cost']
    irrigation_cost_acre = int(bench['irrigation_cost'] * irrigation_multiplier)
    pest_cost_acre = bench['pest_cost']
    machinery_cost_acre = int(bench['machinery_cost'] * machinery_multiplier)
    labor_cost_acre = int(bench['labor_cost'] * labor_multiplier)

    total_cost_per_acre = seed_cost_acre + fert_cost_acre + irrigation_cost_acre + pest_cost_acre + machinery_cost_acre + labor_cost_acre
    total_budget = int(total_cost_per_acre * acres)

    expected_yield_total = round(bench['yield_per_acre'] * acres, 1)
    projected_revenue = int(expected_yield_total * bench['mandi_price'])
    net_profit = projected_revenue - total_budget
    roi_percent = round((net_profit / total_budget) * 100, 1) if total_budget > 0 else 0

    breakdown = {
        'seed': int(seed_cost_acre * acres),
        'fertilizer': int(fert_cost_acre * acres),
        'irrigation': int(irrigation_cost_acre * acres),
        'pesticide': int(pest_cost_acre * acres),
        'machinery': int(machinery_cost_acre * acres),
        'labor': int(labor_cost_acre * acres)
    }

    budget_result = {
        'crop_name': bench['name'],
        'crop_key': crop_key,
        'acres': acres,
        'irrigation_type': irrigation_type,
        'labor_mode': labor_mode,
        'tillage_mode': tillage_mode,
        'mandi_price': bench['mandi_price'],
        'yield_per_acre': bench['yield_per_acre'],
        'expected_yield_total': expected_yield_total,
        'total_budget': total_budget,
        'cost_per_acre': total_cost_per_acre,
        'projected_revenue': projected_revenue,
        'net_profit': net_profit,
        'roi_percent': roi_percent,
        'breakdown': breakdown
    }

    return render_template('budget.html', title=title, res=budget_result, benchmarks=BUDGET_BENCHMARKS)


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)
