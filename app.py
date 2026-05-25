# =============================================================================
# FLASK API — Heart Disease Prediction
# Run: python app.py  →  http://127.0.0.1:5000/
# =============================================================================

import os
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder='templates', static_folder='static')

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
FEATURE_NAMES  = ['age','sex','cp','trestbps','chol','fbs','restecg',
                  'thalach','exang','oldpeak','slope','ca','thal']

# ── Load model + scaler bundle ─────────────────────────────────────────────
try:
    model         = joblib.load(os.path.join(BASE_DIR, 'model.pkl'))
    scaler_bundle = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    MODEL_LOADED  = True
    print("✓  model.pkl  and  scaler.pkl  loaded successfully")
except FileNotFoundError:
    model = scaler_bundle = None
    MODEL_LOADED = False
    print("⚠  model.pkl / scaler.pkl not found — run heart_disease_prediction.py first!")


def _transform(values):
    """Apply imputer → scaler exactly as in training."""
    X = np.array(values, dtype=float).reshape(1, -1)
    if isinstance(scaler_bundle, dict):
        X = scaler_bundle['imputer'].transform(X)
        X = scaler_bundle['scaler'].transform(X)
    else:                        # legacy plain StandardScaler
        X = scaler_bundle.transform(X)
    return X


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':       'ok',
        'model_loaded': MODEL_LOADED,
        'version':      '1.0.0',
        'description':  'Heart Disease Prediction API',
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    Body (JSON):
    {
      "age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,
      "restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":6
    }
    """
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded. Run heart_disease_prediction.py first.'}), 503

    try:
        data    = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No JSON body provided'}), 400

        missing = [f for f in FEATURE_NAMES if f not in data]
        if missing:
            return jsonify({'error': f'Missing features: {missing}'}), 400

        values      = [float(data[f]) for f in FEATURE_NAMES]
        X_scaled    = _transform(values)
        prediction  = int(model.predict(X_scaled)[0])
        probability = float(model.predict_proba(X_scaled)[0][1])

        risk = 'Low' if probability < 0.35 else ('Medium' if probability < 0.65 else 'High')

        factors = []
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
            for i in np.argsort(imp)[::-1][:5]:
                factors.append({'feature': FEATURE_NAMES[i],
                                'value':      round(values[i], 2),
                                'importance': round(float(imp[i]), 4)})

        return jsonify({
            'prediction':  prediction,
            'label':       'Heart Disease Detected' if prediction == 1 else 'No Heart Disease',
            'probability': round(probability * 100, 2),
            'risk_level':  risk,
            'confidence':  round(max(probability, 1 - probability) * 100, 2),
            'top_factors': factors,
        })

    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


# ── Run ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n❤️  HeartGuard AI — Flask API")
    print("   GET  http://127.0.0.1:5000/health")
    print("   POST http://127.0.0.1:5000/predict\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
