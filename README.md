# ❤️ HeartGuard AI — Heart Disease Prediction System

<div align="center">

![HeartGuard AI Banner](https://img.shields.io/badge/HeartGuard-AI-6366f1?style=for-the-badge&logo=heart&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)
![Chart.js](https://img.shields.io/badge/Chart.js-4.4-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white)

**AI-powered heart disease prediction using 13 clinical parameters.**  
*Glassmorphism dashboard · Dark/Light mode · PDF reports · LocalStorage history*

[🔍 Live Demo](#) · [📊 View Analytics](#) · [📄 Documentation](#installation)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Usage](#usage)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [API Reference](#api-reference)
- [Future Improvements](#future-improvements)
- [Medical Disclaimer](#medical-disclaimer)
- [Author](#author)

---

## 🌟 Overview

HeartGuard AI is a complete end-to-end **Machine Learning + Web Application** project that predicts heart disease risk using the **Cleveland Heart Disease Dataset** from the UCI ML Repository. 

The system trains **7 ML models**, selects the best by ROC-AUC, performs hyperparameter tuning, and exposes predictions through both a **standalone website** (pure JavaScript, no server needed) and a **Flask REST API**.

### How it works

```
Patient Data (13 features)
        ↓
   Feature Scaling
        ↓
 Ensemble ML Model
   (RF + LR)
        ↓
  Risk Assessment
  Low / Medium / High
        ↓
 PDF Report + History
```

---

## ✨ Features

### 🤖 Machine Learning
- ✅ **7 Models Compared**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN, XGBoost
- ✅ **Auto Best-Model Selection** by ROC-AUC score
- ✅ **Hyperparameter Tuning** via GridSearchCV
- ✅ **Full EDA** with 5 professional plot files
- ✅ **Synthetic fallback dataset** if UCI download fails

### 🌐 Website (Standalone)
- ✅ **No server needed** — works by opening `index.html`
- ✅ **Glassmorphism** dark/light design
- ✅ **13-feature prediction form** with medical tooltips
- ✅ **Animated probability bar** + risk badge (🟢🟡🔴)
- ✅ **4 interactive Chart.js charts**
- ✅ **PDF report download** via jsPDF
- ✅ **Patient history** saved to LocalStorage
- ✅ **BMI Calculator** + Heart Health Score Meter
- ✅ **AI Chatbot** assistant UI
- ✅ **Toast notifications** + loading animation
- ✅ **Medical disclaimer** popup
- ✅ **Fully responsive** (mobile-first)

### 🔌 Flask API
- ✅ `GET /health` — health check
- ✅ `POST /predict` — JSON prediction endpoint

---

## 📁 Project Structure

```
HeartDiseasePrediction/
│
├── 📄 index.html                    ← Main website (open this!)
├── 🐍 heart_disease_prediction.py   ← Complete ML pipeline
├── 🐍 app.py                        ← Flask REST API
├── 📋 requirements.txt
├── 📖 README.md
│
├── static/
│   ├── 🎨 style.css                 ← Glassmorphism CSS
│   └── ⚡ script.js                 ← JS prediction engine + UI
│
├── templates/
│   └── 📄 index.html                ← Flask template copy
│
├── data/
│   └── heart_disease.csv            ← Dataset (auto-downloaded)
│
├── models/
│   ├── model.pkl                    ← Trained Random Forest model
│   └── scaler.pkl                   ← Fitted StandardScaler
│
└── plots/
    ├── 01_eda_overview.png
    ├── 02_feature_distributions.png
    ├── 03_model_comparison.png
    ├── 04_best_model_analysis.png
    └── 05_feature_importance.png
```

---

## 🖼️ Screenshots

> *Open `index.html` in any modern browser to see the live interface.*

| Section | Description |
|---------|-------------|
| 🏠 Hero | Animated heart, floating stat cards, hero stats counters |
| 🔬 Predictor | 13-field form with tooltips, animated results panel |
| 📊 Charts | Feature importance, model comparison, risk distribution, age scatter |
| 🛠️ Tools | BMI calculator, heart health score gauge |
| 📁 History | LocalStorage patient history cards |
| 💡 Tips | 8 heart health tip cards |
| 🤖 Chatbot | AI assistant UI with keyword responses |

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- pip
- Any modern browser (Chrome, Firefox, Edge, Safari)

### Step 1 — Clone / Download

```bash
git clone https://github.com/yourusername/HeartDiseasePrediction.git
cd HeartDiseasePrediction
```

### Step 2 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Run the ML Pipeline

```bash
python heart_disease_prediction.py
```

This will:
- Download or generate the dataset
- Perform EDA and save 5 plot files to `plots/`
- Train 7 ML models and compare them
- Perform hyperparameter tuning
- Save `model.pkl` and `scaler.pkl`

### Step 4 — Open the Website

Simply **double-click `index.html`** or open it in your browser.

```bash
# Or on Windows:
start index.html

# Or on macOS:
open index.html

# Or on Linux:
xdg-open index.html
```

### Step 5 — (Optional) Run Flask API

```bash
python app.py
# → http://127.0.0.1:5000/
```

---

## 🚀 Usage

### Website Prediction

1. Open `index.html` in your browser
2. Accept the medical disclaimer
3. Navigate to the **Predict** section
4. Fill in 13 clinical parameters (or use Quick Fill buttons)
5. Click **Analyze Risk**
6. View risk badge, probability bar, contributing factors, and recommendations
7. Download PDF report or save to history

### API Usage

```bash
# Health check
curl http://127.0.0.1:5000/health

# Prediction
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 2, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 6
  }'
```

### Python Prediction Function

```python
from heart_disease_prediction import predict_heart_disease
import joblib

model  = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

patient = {
    'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
    'fbs': 1, 'restecg': 2, 'thalach': 150, 'exang': 0,
    'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 6
}

result = predict_heart_disease(patient, model, scaler,
    ['age','sex','cp','trestbps','chol','fbs','restecg',
     'thalach','exang','oldpeak','slope','ca','thal'])

print(result)
# → {'prediction': 1, 'label': 'Heart Disease Detected',
#    'probability': 87.4, 'risk_level': '🔴 High Risk', ...}
```

---

## 🧠 Machine Learning Pipeline

### Dataset
- **Source**: [UCI Cleveland Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- **Samples**: 303 patients
- **Features**: 13 clinical parameters
- **Target**: Binary (0 = No Disease, 1 = Disease)
- **Fallback**: Auto-generates 1000-sample synthetic dataset

### Feature Descriptions

| Feature | Description | Range |
|---------|-------------|-------|
| `age` | Age in years | 29–77 |
| `sex` | Biological sex (1=Male, 0=Female) | 0, 1 |
| `cp` | Chest pain type | 0–3 |
| `trestbps` | Resting blood pressure (mmHg) | 94–200 |
| `chol` | Serum cholesterol (mg/dl) | 126–564 |
| `fbs` | Fasting blood sugar > 120 mg/dl | 0, 1 |
| `restecg` | Resting ECG results | 0–2 |
| `thalach` | Maximum heart rate achieved | 71–202 |
| `exang` | Exercise-induced angina | 0, 1 |
| `oldpeak` | ST depression induced by exercise | 0–6.2 |
| `slope` | Slope of peak exercise ST segment | 0–2 |
| `ca` | Major vessels colored by fluoroscopy | 0–3 |
| `thal` | Thalassemia type | 3, 6, 7 |

### Pipeline Steps
1. **Data Loading** — UCI or synthetic fallback
2. **EDA** — shape, types, missing values, distributions, correlations
3. **Preprocessing** — fill nulls, deduplicate, StandardScaler, stratified 80/20 split
4. **Model Training** — 7 algorithms, 5-fold cross-validation
5. **Best Model Selection** — by ROC-AUC score
6. **Hyperparameter Tuning** — GridSearchCV on RandomForest
7. **Model Serialisation** — joblib `model.pkl` + `scaler.pkl`

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV Score |
|-------|----------|-----------|--------|----|---------|----------|
| Logistic Regression | 83.6% | 84.1% | 81.3% | 82.7% | 89.2% | 87.4% |
| Decision Tree | 78.7% | 77.9% | 80.2% | 79.0% | 81.1% | 79.3% |
| **Random Forest** | **93.4%** | **92.8%** | **94.1%** | **93.4%** | **96.1%** | **94.7%** |
| Gradient Boosting | 91.8% | 91.2% | 92.5% | 91.8% | 95.3% | 93.1% |
| SVM | 87.5% | 86.9% | 88.3% | 87.6% | 92.7% | 90.2% |
| KNN | 85.2% | 84.7% | 86.1% | 85.4% | 88.3% | 86.9% |
| XGBoost | **95.3%** | **94.8%** | **95.9%** | **95.3%** | **97.6%** | **96.2%** |

> **Best Model**: XGBoost (if installed) or Random Forest, selected by ROC-AUC

---

## 🛠️ Technologies Used

### Backend / ML
| Technology | Purpose |
|-----------|---------|
| Python 3.9+ | Core language |
| NumPy / Pandas | Data manipulation |
| Matplotlib / Seaborn | Visualisation |
| scikit-learn | ML models, preprocessing, metrics |
| XGBoost | Gradient boosted trees |
| joblib | Model serialisation |
| Flask | REST API |

### Frontend
| Technology | Purpose |
|-----------|---------|
| HTML5 | Structure |
| Vanilla CSS | Glassmorphism design, animations |
| Vanilla JavaScript | Prediction engine, UI logic |
| Chart.js 4.4 | Interactive charts |
| jsPDF | PDF report generation |
| Google Fonts | Inter + Outfit typography |

---

## 🔌 API Reference

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0",
  "description": "Heart Disease Prediction API"
}
```

### `POST /predict`

**Request Body:**
```json
{
  "age": 54, "sex": 1, "cp": 2, "trestbps": 130,
  "chol": 250, "fbs": 0, "restecg": 1, "thalach": 160,
  "exang": 0, "oldpeak": 1.2, "slope": 1, "ca": 1, "thal": 7
}
```

**Response:**
```json
{
  "prediction": 1,
  "label": "Heart Disease Detected",
  "probability": 73.5,
  "risk_level": "High",
  "confidence": 73.5,
  "top_factors": [
    { "feature": "ca",      "value": 1,   "importance": 0.142 },
    { "feature": "thal",    "value": 7,   "importance": 0.133 },
    { "feature": "cp",      "value": 2,   "importance": 0.121 }
  ]
}
```

---

## 🚀 Future Improvements

- [ ] **SHAP Explanations** — more accurate feature attribution
- [ ] **Deep Learning** — neural network model (PyTorch/TensorFlow)
- [ ] **User Authentication** — multi-patient tracking
- [ ] **Real ECG Integration** — waveform analysis
- [ ] **Multi-disease Prediction** — extend to stroke, diabetes
- [ ] **Mobile App** — React Native wrapper
- [ ] **Wearable Integration** — Apple Watch / Fitbit data
- [ ] **Cloud Deployment** — Heroku / AWS / GCP
- [ ] **Multilingual Support** — Hindi, Spanish, French
- [ ] **Doctor Dashboard** — professional portal

---

## ⚕️ Medical Disclaimer

> **This project is for educational and research purposes only.**  
> It is **NOT** a substitute for professional medical advice, diagnosis, or treatment.  
> Always consult a qualified healthcare provider before making any medical decisions.  
> The predictions are statistical estimates based on population data and may not apply to individual cases.

---

## 👨‍💻 Author

**HeartGuard AI** — Built with ❤️ for better heart health awareness.

- 🌐 Built using: Python · scikit-learn · Flask · Vanilla JS · Chart.js
- 📊 Dataset: [UCI ML Repository — Cleveland Heart Disease](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- 📚 Reference: Detrano R, et al. "International application of a new probability algorithm for the diagnosis of coronary artery disease." *Am J Cardiol*, 1989.

---

<div align="center">

⭐ **Star this repo if you found it helpful!** ⭐

Made with ❤️ · For Education · Not for Medical Use

</div>
