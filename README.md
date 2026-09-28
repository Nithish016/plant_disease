# 🌾 Kisan AI - Precision Agriculture & Market Intelligence Platform
#### Developed by **Gubba Nithish** 

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask%203.x-green.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-red.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Auth](https://img.shields.io/badge/Auth-Clerk%20Dynamic-purple.svg)](https://clerk.com/)

**Kisan AI** is a comprehensive environmental intelligence, precision farming, and agricultural economics decision support system designed to empower farmers and agribusinesses with actionable, AI-driven insights across every stage of the crop lifecycle.

---

## 🌟 Core Features & Modules

### 1. 🌐 Bi-lingual Support (English / हिन्दी)
- Instant, persistent dual-language switching across all pages with a single click.
- Seamlessly accessible for regional farmers and agri-specialists alike.

### 2. 📈 Live Mandi & Agricultural Stock Market Prices (`/market`)
- Real-time price tracking for key commodities across major wholesale mandis (Wheat, Paddy, Cotton, Soybean, Mustard, Maize, Tomato, Potato, Desi Chana, Sugarcane).
- 24-hour price swings, high/low intraday bands, Government MSP comparison, and arrival volumes (Quintals).
- Live real-time search & filter bar in both English and Hindi.

### 3. 💰 Farm Cultivation Budget & Profit Predictor (`/budget-predictor`)
- Calculate total cultivation expenses per acre (certified seeds, fertilizers, irrigation, plant protection sprays, tillage/machinery, and labor).
- Project gross mandi revenue, net profit, and Return on Investment (ROI %).
- Includes actionable precision agriculture tips for water/fertilizer cost savings and profit maximization.

### 4. 🌱 Crop Recommendation System (`/crop-recommend`)
- Predicts the most suitable crop based on soil Nitrogen (N), Phosphorous (P), Potassium (K), soil pH, and real-time city temperature, humidity, and rainfall fetched via OpenWeather API.

### 5. 🧪 Fertilizer & Nutrient Suggestion (`/fertilizer`)
- Analyzes soil nutrient deficiencies or excesses relative to target crop standards and provides tailored inorganic and organic fertilizer recommendations.

### 6. 🔬 Deep Learning Plant Disease Diagnosis (`/disease-prediction`)
- Upload leaf photographs to instantly detect 38+ plant diseases using custom PyTorch ResNet convolutional neural network architectures with detailed prevention and remediation guidelines.

### 7. 🔑 Dynamic User Authentication (`/login`)
- Powered by Clerk JS SDK with pre-configured agri-architect profiles, guest logins, and secure session management.

---

## 🛠️ Technology Stack
- **Backend**: Python 3.10+, Flask 3.x, MarkupSafe, Requests
- **Machine Learning & AI**: PyTorch, Torchvision, Scikit-Learn, NumPy, Pandas
- **Frontend**: Responsive HTML5, Bootstrap 4, Vanilla CSS, FontAwesome, JavaScript
- **APIs & Auth**: OpenWeatherMap API, Clerk Authentication SDK

---

## 🚀 Getting Started Locally

### Prerequisites
- Python 3.10+ (Recommended: Python 3.11 / 3.14)
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/Nithish016/plant_disease.git
cd plant_disease/app

# Install dependencies
pip install -r requirements.txt
pip install torch torchvision
```

### Running Kisan AI
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

---

## 👤 Author & Credits
- **Developed by**: **Gubba Nithish** ❤️
- **Project**: Kisan AI (Precision Agriculture Platform)
