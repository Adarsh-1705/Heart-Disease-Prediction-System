# -*- coding: utf-8 -*-
# =============================================================================
# HEART DISEASE PREDICTION SYSTEM
# Complete ML Pipeline: EDA -> Preprocessing -> Training -> Tuning -> Prediction
# =============================================================================

import io
import sys
import os
import warnings

# Force UTF-8 output on Windows (fixes cp1252 emoji errors)
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, roc_curve)

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR  = os.path.join(BASE_DIR, 'plots')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR   = os.path.join(BASE_DIR, 'data')

for d in [PLOTS_DIR, MODELS_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# STYLE
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'figure.facecolor': '#0f172a',
    'axes.facecolor':   '#1e293b',
    'axes.edgecolor':   '#334155',
    'axes.labelcolor':  '#e2e8f0',
    'xtick.color':      '#94a3b8',
    'ytick.color':      '#94a3b8',
    'text.color':       '#e2e8f0',
    'grid.color':       '#334155',
    'grid.alpha':       0.5,
    'font.family':      'DejaVu Sans',
    'font.size':        10,
})

PALETTE = ['#6366f1', '#ec4899', '#10b981', '#f59e0b', '#3b82f6', '#ef4444', '#8b5cf6']

# =============================================================================
# STEP 0 — LOAD DATA
# =============================================================================

def load_dataset():
    """
    Try to load the Cleveland Heart Disease dataset.
    Falls back to a synthetic dataset if download fails.
    """
    print("\n" + "="*60)
    print("  [*] LOADING DATASET")
    print("="*60)

    columns = ['age','sex','cp','trestbps','chol','fbs','restecg',
               'thalach','exang','oldpeak','slope','ca','thal','target']

    # --- Try UCI URL ---
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "heart-disease/processed.cleveland.data")
    try:
        df = pd.read_csv(url, header=None, names=columns, na_values='?')
        df['target'] = (df['target'] > 0).astype(int)
        df.to_csv(os.path.join(DATA_DIR, 'heart_disease.csv'), index=False)
        print(f"  [OK] Dataset loaded from UCI Repository  ({df.shape[0]} rows)")
        return df
    except Exception as e:
        print(f"  [!] UCI download failed ({e}). Generating synthetic dataset...")

    # --- Synthetic fallback ---
    from sklearn.datasets import make_classification
    np.random.seed(42)
    n = 1000
    X, y = make_classification(n_samples=n, n_features=13, n_informative=9,
                                n_redundant=2, random_state=42, weights=[0.55, 0.45])
    feature_names = columns[:-1]
    df = pd.DataFrame(X, columns=feature_names)

    # Make features look realistic
    df['age']     = (df['age']     * 10 + 55).clip(29, 77).round()
    df['sex']     = (df['sex']     > 0).astype(int)
    df['cp']      = np.random.randint(0, 4, n)
    df['trestbps']= (df['trestbps']* 15 + 131).clip(94, 200).round()
    df['chol']    = (df['chol']    * 40 + 246).clip(126, 564).round()
    df['fbs']     = (df['fbs']     > 1).astype(int)
    df['restecg'] = np.random.randint(0, 3, n)
    df['thalach'] = (df['thalach'] * 20 + 149).clip(71, 202).round()
    df['exang']   = (df['exang']   > 0.5).astype(int)
    df['oldpeak'] = (df['oldpeak'] * 0.8 + 1.0).clip(0, 6.2).round(1)
    df['slope']   = np.random.randint(0, 3, n)
    df['ca']      = np.random.randint(0, 4, n)
    df['thal']    = np.random.choice([3, 6, 7], n)
    df['target']  = y

    df.to_csv(os.path.join(DATA_DIR, 'heart_disease.csv'), index=False)
    print(f"  [OK] Synthetic dataset generated  ({df.shape[0]} rows)")
    return df


# =============================================================================
# STEP 1 — EXPLORATORY DATA ANALYSIS
# =============================================================================

def perform_eda(df):
    print("\n" + "="*60)
    print("  STEP 1 -- EXPLORATORY DATA ANALYSIS")
    print("="*60)

    print(f"\n  Shape         : {df.shape}")
    print(f"  Columns       : {list(df.columns)}")
    print(f"\n  Data Types:\n{df.dtypes}")
    print(f"\n  Missing Values:\n{df.isnull().sum()}")
    print(f"\n  Duplicates    : {df.duplicated().sum()}")
    print(f"\n  Statistical Summary:\n{df.describe().round(2)}")
    print(f"\n  Target Distribution:\n{df['target'].value_counts()}")
    print(f"   -> No Disease : {(df['target']==0).sum()} ({(df['target']==0).mean()*100:.1f}%)")
    print(f"   -> Disease    : {(df['target']==1).sum()} ({(df['target']==1).mean()*100:.1f}%)")

    # ---------------------------------------------------------------
    # PLOT 01 — EDA Overview (4 charts)
    # ---------------------------------------------------------------
    fig = plt.figure(figsize=(20, 16), facecolor='#0f172a')
    fig.suptitle('Heart Disease Dataset -- EDA Overview',
                 fontsize=22, fontweight='bold', color='#f1f5f9', y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # 1a — Target distribution
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df['target'].value_counts()
    bars = ax1.bar(['No Disease', 'Disease'], counts.values,
                   color=[PALETTE[2], PALETTE[1]], width=0.5, edgecolor='#1e293b', linewidth=2)
    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                 f'{val}\n({val/len(df)*100:.1f}%)',
                 ha='center', fontsize=11, fontweight='bold', color='#e2e8f0')
    ax1.set_title('Target Distribution', fontsize=14, fontweight='bold', color='#f1f5f9', pad=12)
    ax1.set_ylabel('Count', color='#94a3b8')
    ax1.set_ylim(0, max(counts.values) * 1.25)
    ax1.spines[['top','right']].set_visible(False)

    # 1b — Age distribution by target
    ax2 = fig.add_subplot(gs[0, 1])
    for tgt, label, color in zip([0,1], ['No Disease','Disease'], [PALETTE[2], PALETTE[1]]):
        subset = df[df['target']==tgt]['age']
        ax2.hist(subset, bins=15, alpha=0.75, label=label, color=color, edgecolor='#0f172a')
    ax2.set_title('Age Distribution by Target', fontsize=14, fontweight='bold', color='#f1f5f9', pad=12)
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Count')
    ax2.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0')
    ax2.spines[['top','right']].set_visible(False)

    # 1c — Correlation heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    sns.heatmap(corr, ax=ax3, mask=mask, cmap=cmap, center=0,
                annot=True, fmt='.2f', annot_kws={'size':7},
                linewidths=0.5, linecolor='#0f172a',
                cbar_kws={'shrink': 0.8})
    ax3.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', color='#f1f5f9', pad=12)
    ax3.tick_params(axis='x', rotation=45)

    # 1d — Cholesterol by target (boxplot)
    ax4 = fig.add_subplot(gs[1, 1])
    data_no  = df[df['target']==0]['chol'].dropna()
    data_yes = df[df['target']==1]['chol'].dropna()
    bp = ax4.boxplot([data_no, data_yes], patch_artist=True,
                     widths=0.4, notch=False,
                     boxprops=dict(linewidth=1.5),
                     medianprops=dict(color='#fbbf24', linewidth=2),
                     whiskerprops=dict(color='#94a3b8'),
                     capprops=dict(color='#94a3b8'),
                     flierprops=dict(marker='o', markersize=4, alpha=0.5))
    bp['boxes'][0].set_facecolor(PALETTE[2])
    bp['boxes'][1].set_facecolor(PALETTE[1])
    ax4.set_xticklabels(['No Disease', 'Disease'])
    ax4.set_title('Cholesterol Distribution by Target', fontsize=14, fontweight='bold', color='#f1f5f9', pad=12)
    ax4.set_ylabel('Cholesterol (mg/dl)')
    ax4.spines[['top','right']].set_visible(False)

    out = os.path.join(PLOTS_DIR, '01_eda_overview.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    print(f"\n  [OK] Saved: {out}")

    # ---------------------------------------------------------------
    # PLOT 02 — Feature Distributions
    # ---------------------------------------------------------------
    features = ['age','trestbps','chol','thalach','oldpeak','cp','ca','slope','thal']
    fig, axes = plt.subplots(3, 3, figsize=(20, 14), facecolor='#0f172a')
    fig.suptitle('Feature Distributions by Target',
                 fontsize=20, fontweight='bold', color='#f1f5f9', y=1.01)

    for ax, feat in zip(axes.flat, features):
        for tgt, label, color in zip([0,1], ['No Disease','Disease'], [PALETTE[2], PALETTE[1]]):
            vals = df[df['target']==tgt][feat].dropna()
            ax.hist(vals, bins=20, alpha=0.7, label=label, color=color, edgecolor='#0f172a')
        ax.set_title(feat.upper(), fontsize=12, fontweight='bold', color='#a78bfa')
        ax.set_xlabel(feat, color='#94a3b8', fontsize=9)
        ax.set_ylabel('Count', color='#94a3b8', fontsize=9)
        ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8)
        ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, '02_feature_distributions.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    print(f"  [OK] Saved: {out}")


# =============================================================================
# STEP 2 — PREPROCESSING
# =============================================================================

def preprocess(df):
    print("\n" + "="*60)
    print("  STEP 2 -- DATA PREPROCESSING")
    print("="*60)

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  [OK] Removed {before - len(df)} duplicate rows")

    # Count missing before filling
    missing_before = int(df.isnull().sum().sum())

    # Features / Target
    X = df.drop('target', axis=1)
    y = df['target']
    feature_names = X.columns.tolist()

    # Train-test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Impute NaNs (ca/thal have 4 and 2 missing) then standardise
    from sklearn.impute import SimpleImputer
    imputer   = SimpleImputer(strategy='median')
    scaler    = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)
    X_train_sc  = scaler.fit_transform(X_train_imp)
    X_test_sc   = scaler.transform(X_test_imp)

    print(f"  [OK] Imputed {missing_before} missing values (median strategy)")
    print(f"  [OK] Train size : {X_train.shape}  |  Test size: {X_test.shape}")

    # Save scaler bundle
    scaler_bundle = {'imputer': imputer, 'scaler': scaler}
    scaler_path   = os.path.join(MODELS_DIR, 'scaler.pkl')
    joblib.dump(scaler_bundle, scaler_path)
    joblib.dump(scaler_bundle, os.path.join(BASE_DIR, 'scaler.pkl'))
    print(f"  [OK] Scaler saved -> {scaler_path}")

    return X_train_sc, X_test_sc, y_train, y_test, scaler_bundle, feature_names, X, y


# =============================================================================
# STEP 3 — TRAIN & COMPARE MODELS
# =============================================================================

def train_models(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print("  STEP 3 -- MODEL TRAINING & COMPARISON")
    print("="*60)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(random_state=42),
        'SVM':                 SVC(probability=True, random_state=42),
        'KNN':                 KNeighborsClassifier(),
    }

    # Try XGBoost
    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0)
        print("  [OK] XGBoost included")
    except ImportError:
        print("  [!] XGBoost not installed -- skipping")

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_prob  = model.predict_proba(X_test)[:,1]
        cv_score= cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()

        results[name] = {
            'model':     model,
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall':    recall_score(y_test, y_pred, zero_division=0),
            'f1':        f1_score(y_test, y_pred, zero_division=0),
            'roc_auc':   roc_auc_score(y_test, y_prob),
            'cv_score':  cv_score,
            'y_pred':    y_pred,
            'y_prob':    y_prob,
        }
        print(f"  [OK] {name:25s}  Acc={results[name]['accuracy']:.4f}  "
              f"AUC={results[name]['roc_auc']:.4f}  CV={cv_score:.4f}")

    # -----------------------------------------------------------------
    # PLOT 03 — Model Comparison
    # -----------------------------------------------------------------
    metrics = ['accuracy','precision','recall','f1','roc_auc','cv_score']
    labels  = ['Accuracy','Precision','Recall','F1','ROC-AUC','CV Score']
    names   = list(results.keys())
    x = np.arange(len(names))

    fig, axes = plt.subplots(2, 3, figsize=(22, 12), facecolor='#0f172a')
    fig.suptitle('Model Comparison Dashboard',
                 fontsize=20, fontweight='bold', color='#f1f5f9', y=1.01)

    for ax, metric, label in zip(axes.flat, metrics, labels):
        vals = [results[n][metric] for n in names]
        bars = ax.bar(names, vals, color=PALETTE[:len(names)], edgecolor='#0f172a', linewidth=1.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color='#e2e8f0')
        ax.set_title(label, fontsize=13, fontweight='bold', color='#a78bfa')
        ax.set_ylim(0, 1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        ax.axhline(0.8, color='#fbbf24', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, '03_model_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    print(f"\n  [OK] Saved: {out}")

    return results


# =============================================================================
# STEP 4 — BEST MODEL ANALYSIS
# =============================================================================

def analyze_best_model(results, X_test, y_test, feature_names):
    print("\n" + "="*60)
    print("  STEP 4 -- BEST MODEL ANALYSIS")
    print("="*60)

    best_name  = max(results, key=lambda n: results[n]['roc_auc'])
    best       = results[best_name]
    best_model = best['model']
    y_pred     = best['y_pred']
    y_prob     = best['y_prob']

    print(f"\n  [BEST] Best model : {best_name}")
    print(f"         ROC-AUC   : {best['roc_auc']:.4f}")
    print(f"         Accuracy  : {best['accuracy']:.4f}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=['No Disease','Disease']))

    # -----------------------------------------------------------------
    # PLOT 04 — Best Model Analysis (Confusion Matrix + ROC + Prob)
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor='#0f172a')
    fig.suptitle(f'Best Model: {best_name}',
                 fontsize=18, fontweight='bold', color='#f1f5f9', y=1.03)

    # 4a — Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, ax=axes[0], annot=True, fmt='d', cmap='RdYlGn',
                linewidths=2, linecolor='#0f172a',
                xticklabels=['No Disease','Disease'],
                yticklabels=['No Disease','Disease'],
                annot_kws={'size':16, 'fontweight':'bold'})
    axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold', color='#f1f5f9', pad=12)
    axes[0].set_xlabel('Predicted', color='#94a3b8')
    axes[0].set_ylabel('Actual', color='#94a3b8')

    # 4b — ROC Curves for all models
    for (name, res), color in zip(results.items(), PALETTE):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        axes[1].plot(fpr, tpr, color=color, linewidth=2,
                     label=f"{name} (AUC={res['roc_auc']:.3f})")
    axes[1].plot([0,1],[0,1], 'w--', linewidth=1, alpha=0.5)
    axes[1].set_title('ROC Curves — All Models', fontsize=14, fontweight='bold', color='#f1f5f9', pad=12)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8)
    axes[1].spines[['top','right']].set_visible(False)

    # 4c — Probability distribution
    axes[2].hist(y_prob[y_test==0], bins=25, alpha=0.75, color=PALETTE[2], label='No Disease', edgecolor='#0f172a')
    axes[2].hist(y_prob[y_test==1], bins=25, alpha=0.75, color=PALETTE[1], label='Disease', edgecolor='#0f172a')
    axes[2].axvline(0.5, color='#fbbf24', linestyle='--', linewidth=2, label='Threshold=0.5')
    axes[2].set_title('Predicted Probability Distribution', fontsize=14, fontweight='bold', color='#f1f5f9', pad=12)
    axes[2].set_xlabel('Probability of Disease')
    axes[2].set_ylabel('Count')
    axes[2].legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0')
    axes[2].spines[['top','right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, '04_best_model_analysis.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    print(f"\n  [OK] Saved: {out}")

    # -----------------------------------------------------------------
    # PLOT 05 — Feature Importance
    # -----------------------------------------------------------------
    importance = None
    if hasattr(best_model, 'feature_importances_'):
        importance = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importance = np.abs(best_model.coef_[0])

    if importance is not None:
        idx = np.argsort(importance)[::-1]
        sorted_feat = [feature_names[i] for i in idx]
        sorted_imp  = importance[idx]

        fig, ax = plt.subplots(figsize=(14, 8), facecolor='#0f172a')
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(sorted_feat))]
        bars = ax.barh(sorted_feat[::-1], sorted_imp[::-1], color=colors[::-1],
                       edgecolor='#0f172a', linewidth=1.5)
        for bar, val in zip(bars, sorted_imp[::-1]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9, color='#e2e8f0')
        ax.set_title(f'Feature Importance -- {best_name}',
                     fontsize=16, fontweight='bold', color='#f1f5f9', pad=16)
        ax.set_xlabel('Importance Score', color='#94a3b8')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, '05_feature_importance.png')
        plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f172a')
        plt.close()
        print(f"  [OK] Saved: {out}")

    return best_name, best_model


# =============================================================================
# STEP 5 — HYPERPARAMETER TUNING (Random Forest)
# =============================================================================

def tune_random_forest(X_train, y_train):
    print("\n" + "="*60)
    print("  STEP 5 -- HYPERPARAMETER TUNING (RandomForest)")
    print("="*60)

    param_grid = {
        'n_estimators':    [100, 200],
        'max_depth':       [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf':  [1, 2],
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc',
                               n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)

    print(f"  [OK] Best Parameters : {grid_search.best_params_}")
    print(f"  [OK] Best CV AUC     : {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


# =============================================================================
# STEP 6 — PREDICTION SYSTEM
# =============================================================================

def predict_heart_disease(patient_data: dict, model, scaler_bundle, feature_names: list):
    """
    Predict heart disease for a patient.

    Parameters
    ----------
    patient_data  : dict  - 13 feature values keyed by feature name
    model         : trained classifier
    scaler_bundle : dict with keys 'imputer' (SimpleImputer) and 'scaler' (StandardScaler)
    feature_names : list of feature names in order

    Returns
    -------
    dict with prediction, probability, risk_level, confidence, factors
    """
    # Build feature vector
    values   = [patient_data.get(f, 0) for f in feature_names]
    X        = np.array(values, dtype=float).reshape(1, -1)

    # Apply same impute + scale pipeline used during training
    X_imp    = scaler_bundle['imputer'].transform(X)
    X_scaled = scaler_bundle['scaler'].transform(X_imp)

    prediction  = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]

    # Risk level
    if probability < 0.35:
        risk_level = '[LOW]    Low Risk'
    elif probability < 0.65:
        risk_level = '[MEDIUM] Medium Risk'
    else:
        risk_level = '[HIGH]   High Risk'

    # Feature contributions (if tree-based)
    factors = []
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        idx = np.argsort(imp)[::-1][:5]
        for i in idx:
            factors.append({
                'feature':    feature_names[i],
                'value':      round(float(values[i]), 2),
                'importance': round(float(imp[i]), 4),
            })

    return {
        'prediction':  int(prediction),
        'label':       'Heart Disease Detected' if prediction == 1 else 'No Heart Disease',
        'probability': round(float(probability) * 100, 2),
        'risk_level':  risk_level,
        'confidence':  round(max(probability, 1 - probability) * 100, 2),
        'top_factors': factors,
    }


def run_sample_predictions(model, scaler, feature_names):
    print("\n" + "="*60)
    print("  STEP 6 -- SAMPLE PREDICTIONS")
    print("="*60)

    samples = [
        {
            'name': 'Patient A (High Risk Profile)',
            'data': {'age':63,'sex':1,'cp':3,'trestbps':145,'chol':233,'fbs':1,
                     'restecg':2,'thalach':150,'exang':0,'oldpeak':2.3,'slope':0,'ca':0,'thal':6},
        },
        {
            'name': 'Patient B (Low Risk Profile)',
            'data': {'age':41,'sex':0,'cp':1,'trestbps':130,'chol':204,'fbs':0,
                     'restecg':0,'thalach':172,'exang':0,'oldpeak':1.4,'slope':2,'ca':0,'thal':2},
        },
    ]

    for sample in samples:
        result = predict_heart_disease(sample['data'], model, scaler, feature_names)
        print(f"\n  Patient    : {sample['name']}")
        print(f"  Prediction : {result['label']}")
        print(f"  Probability: {result['probability']}%")
        print(f"  Risk Level : {result['risk_level']}")
        print(f"  Confidence : {result['confidence']}%")
        if result['top_factors']:
            print(f"  Top Factors:")
            for f in result['top_factors']:
                print(f"    - {f['feature']} = {f['value']}  (importance: {f['importance']})")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*60)
    print("  HEART DISEASE PREDICTION SYSTEM")
    print("="*60)

    # Load data
    df = load_dataset()

    # EDA
    perform_eda(df)

    # Preprocess
    X_train, X_test, y_train, y_test, scaler, feature_names, X, y = preprocess(df)

    # Train models
    results = train_models(X_train, X_test, y_train, y_test)

    # Best model analysis
    best_name, best_model = analyze_best_model(results, X_test, y_test, feature_names)

    # Hyperparameter tuning
    tuned_model = tune_random_forest(X_train, y_train)

    # Save final model (scaler already saved in preprocess step)
    model_path = os.path.join(MODELS_DIR, 'model.pkl')
    joblib.dump(tuned_model, model_path)
    joblib.dump(tuned_model, os.path.join(BASE_DIR, 'model.pkl'))
    print(f"\n  [OK] Final model saved -> {model_path}")

    # Sample predictions
    run_sample_predictions(tuned_model, scaler, feature_names)

    # Summary
    print("\n" + "="*60)
    print("  [DONE] PIPELINE COMPLETE")
    print("="*60)
    print(f"  Best Model     : {best_name}")
    print(f"  ROC-AUC        : {results[best_name]['roc_auc']:.4f}")
    print(f"  Plots saved    : plots/")
    print(f"  Models saved   : models/")
    print("  -> Open index.html in your browser to use the web app!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
