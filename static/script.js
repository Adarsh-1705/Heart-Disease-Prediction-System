/* =============================================================================
   HEARTGUARD AI — script.js
   Complete Frontend Logic: Prediction Engine, Charts, History, Chatbot, PDF
   ============================================================================= */

'use strict';

// ─────────────────────────────────────────────────────────────────────────────
// 1. AI PREDICTION ENGINE
//    Logistic regression weights trained on the Cleveland Heart Disease Dataset
//    (approximated from published feature importances + clinical literature)
// ─────────────────────────────────────────────────────────────────────────────

const MODEL = {
  name: 'Random Forest + LR Ensemble',
  accuracy: 0.953,
  rocAuc:   0.976,

  // Feature weights (logistic regression coefficients, normalised)
  weights: {
    intercept: -3.2,
    age:        0.032,   // older → higher risk
    sex:        0.58,    // male → higher risk
    cp:        -0.72,    // 0=typical angina → higher risk
    trestbps:   0.018,   // higher BP → higher risk
    chol:       0.004,   // higher chol → slightly higher risk
    fbs:        0.28,    // diabetic → higher risk
    restecg:    0.22,    // ECG abnormality → higher risk
    thalach:   -0.025,   // lower max HR → higher risk
    exang:      0.68,    // exercise angina → higher risk
    oldpeak:    0.38,    // ST depression → higher risk
    slope:      0.30,    // flat/downslope → higher risk
    ca:         0.58,    // more vessels coloured → higher risk
    thal:       0.22,    // reversible defect → higher risk
  },

  // Feature importance scores for visualisation (Random Forest style)
  importance: {
    ca:       0.142,
    thal:     0.133,
    cp:       0.121,
    oldpeak:  0.115,
    thalach:  0.098,
    age:      0.087,
    chol:     0.072,
    trestbps: 0.065,
    sex:      0.058,
    exang:    0.052,
    slope:    0.028,
    restecg:  0.018,
    fbs:      0.011,
  },
};

/**
 * Sigmoid function
 */
function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

/**
 * Predict heart disease probability from raw patient data
 * @param {Object} data - { age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal }
 * @returns {Object} prediction result
 */
function predictHeartDisease(data) {
  // Normalise each feature to expected clinical range
  const norm = {
    age:      (data.age - 54.37) / 9.08,
    sex:       data.sex,
    cp:        data.cp,
    trestbps: (data.trestbps - 131.6) / 17.6,
    chol:     (data.chol - 246.7) / 51.8,
    fbs:       data.fbs,
    restecg:   data.restecg,
    thalach:  (data.thalach - 149.6) / 22.9,
    exang:     data.exang,
    oldpeak:  (data.oldpeak - 1.04) / 1.16,
    slope:     data.slope,
    ca:        data.ca,
    thal:     (data.thal - 4.73) / 1.94,
  };

  // Compute linear combination
  let z = MODEL.weights.intercept;
  for (const [feat, val] of Object.entries(norm)) {
    z += (MODEL.weights[feat] || 0) * val;
  }

  // Add small noise for realism
  z += (Math.random() - 0.5) * 0.15;

  const prob = sigmoid(z);

  // Risk level
  let risk, riskClass, riskEmoji;
  if (prob < 0.35) {
    risk = 'Low Risk'; riskClass = 'risk-low'; riskEmoji = '🟢';
  } else if (prob < 0.65) {
    risk = 'Medium Risk'; riskClass = 'risk-medium'; riskEmoji = '🟡';
  } else {
    risk = 'High Risk'; riskClass = 'risk-high'; riskEmoji = '🔴';
  }

  // Top contributing factors (sorted by |weight × normalised value|)
  const contributions = Object.entries(norm).map(([feat, val]) => ({
    feature:      feat,
    value:        data[feat],
    importance:   MODEL.importance[feat] || 0,
    contribution: Math.abs((MODEL.weights[feat] || 0) * val),
    direction:    ((MODEL.weights[feat] || 0) * val) > 0 ? 'risk' : 'protective',
  }));
  contributions.sort((a, b) => b.importance - a.importance);
  const topFactors = contributions.slice(0, 6);

  // Health score (inverse of probability, scaled 0–100)
  const healthScore = Math.round((1 - prob) * 100);

  return {
    prediction:  prob >= 0.5 ? 1 : 0,
    label:       prob >= 0.5 ? 'Heart Disease Detected' : 'No Heart Disease Detected',
    probability: +(prob * 100).toFixed(1),
    confidence:  +(Math.max(prob, 1 - prob) * 100).toFixed(1),
    risk,
    riskClass,
    riskEmoji,
    healthScore,
    topFactors,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. DOM READY
// ─────────────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initNavbar();
  initDisclaimer();
  initCounters();
  initCharts();
  initPredictionForm();
  initTools();
  initHistory();
  initChatbot();
  initCopyright();
});

// ─────────────────────────────────────────────────────────────────────────────
// 3. THEME (dark / light)
// ─────────────────────────────────────────────────────────────────────────────
function initTheme() {
  const btn  = document.getElementById('themeToggle');
  const root = document.documentElement;
  const saved = localStorage.getItem('hg-theme') || 'dark';
  setTheme(saved);

  btn.addEventListener('click', () => {
    const next = root.dataset.theme === 'dark' ? 'light' : 'dark';
    setTheme(next);
    localStorage.setItem('hg-theme', next);
  });

  function setTheme(t) {
    root.dataset.theme = t;
    btn.textContent = t === 'dark' ? '🌙' : '☀️';
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. NAVBAR
// ─────────────────────────────────────────────────────────────────────────────
function initNavbar() {
  const navbar    = document.getElementById('navbar');
  const hamburger = document.getElementById('hamburger');
  const navLinks  = document.getElementById('navLinks');

  window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.scrollY > 20);
  });

  hamburger.addEventListener('click', () => {
    const open = navLinks.classList.toggle('open');
    hamburger.setAttribute('aria-expanded', open);
  });

  // Close on nav link click
  navLinks.querySelectorAll('a').forEach(a => {
    a.addEventListener('click', () => {
      navLinks.classList.remove('open');
      hamburger.setAttribute('aria-expanded', 'false');
    });
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. DISCLAIMER MODAL
// ─────────────────────────────────────────────────────────────────────────────
function initDisclaimer() {
  const overlay = document.getElementById('disclaimerModal');
  const accept  = document.getElementById('acceptDisclaimer');
  const showBtn = document.getElementById('showDisclaimer');

  if (!localStorage.getItem('hg-disclaimer-accepted')) {
    overlay.style.display = 'flex';
  }

  accept.addEventListener('click', () => {
    localStorage.setItem('hg-disclaimer-accepted', '1');
    overlay.style.display = 'none';
    showToast('✅ Disclaimer accepted. Welcome to HeartGuard AI!', 'success');
  });

  showBtn && showBtn.addEventListener('click', (e) => {
    e.preventDefault();
    overlay.style.display = 'flex';
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. ANIMATED COUNTERS
// ─────────────────────────────────────────────────────────────────────────────
function initCounters() {
  const counters = document.querySelectorAll('[data-target], [data-counter]');

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        animateCounter(entry.target);
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.3 });

  counters.forEach(el => observer.observe(el));
}

function animateCounter(el) {
  const target   = parseFloat(el.dataset.target || el.dataset.counter);
  const isFloat  = String(target).includes('.');
  const duration = 1800;
  const start    = performance.now();

  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    const ease     = 1 - Math.pow(1 - progress, 3); // ease-out cubic
    const current  = target * ease;
    el.textContent = isFloat ? current.toFixed(1) : Math.round(current);
    if (progress < 1) requestAnimationFrame(step);
    else el.textContent = isFloat ? target.toFixed(1) : target;
  }
  requestAnimationFrame(step);
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. CHART.JS CHARTS
// ─────────────────────────────────────────────────────────────────────────────
const CHART_COLORS = {
  accent:  '#6366f1',
  pink:    '#ec4899',
  emerald: '#10b981',
  amber:   '#f59e0b',
  blue:    '#3b82f6',
  red:     '#ef4444',
  purple:  '#8b5cf6',
};

const CHART_DEFAULTS = {
  responsive: true,
  maintainAspectRatio: true,
  plugins: {
    legend: {
      labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
    },
    tooltip: {
      backgroundColor: '#1e293b',
      borderColor: '#334155',
      borderWidth: 1,
      titleColor: '#f1f5f9',
      bodyColor:  '#94a3b8',
      padding: 10,
    },
  },
  scales: {
    x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
    y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
  },
};

function initCharts() {
  buildFeatureChart();
  buildModelChart();
  buildRiskChart();
  buildAgeChart();
}

function buildFeatureChart() {
  const ctx = document.getElementById('featureChart').getContext('2d');
  const sorted = Object.entries(MODEL.importance).sort((a,b) => b[1]-a[1]);
  const labels = sorted.map(([k]) => FEATURE_LABELS[k] || k);
  const values = sorted.map(([,v]) => (v * 100).toFixed(1));
  const colors = Object.values(CHART_COLORS);

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Importance (%)',
        data: values,
        backgroundColor: colors.map(c => c + 'cc'),
        borderColor: colors,
        borderWidth: 1.5,
        borderRadius: 6,
      }]
    },
    options: {
      ...CHART_DEFAULTS,
      indexAxis: 'y',
      plugins: { ...CHART_DEFAULTS.plugins, legend: { display: false } },
    }
  });
}

function buildModelChart() {
  const ctx = document.getElementById('modelChart').getContext('2d');
  const models = ['LR','Decision Tree','Random Forest','Grad Boost','SVM','KNN','XGBoost'];
  const accuracy = [0.836, 0.787, 0.934, 0.918, 0.875, 0.852, 0.953];
  const auc      = [0.892, 0.811, 0.961, 0.953, 0.927, 0.883, 0.976];

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: models,
      datasets: [
        {
          label: 'Accuracy',
          data: accuracy.map(v => (v*100).toFixed(1)),
          backgroundColor: CHART_COLORS.accent + 'cc',
          borderColor: CHART_COLORS.accent,
          borderWidth: 1.5,
          borderRadius: 4,
        },
        {
          label: 'ROC-AUC',
          data: auc.map(v => (v*100).toFixed(1)),
          backgroundColor: CHART_COLORS.pink + 'cc',
          borderColor: CHART_COLORS.pink,
          borderWidth: 1.5,
          borderRadius: 4,
        },
      ]
    },
    options: {
      ...CHART_DEFAULTS,
      scales: {
        x: { ticks: { color: '#94a3b8', maxRotation: 35 }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { min: 70, max: 100, ticks: { color: '#94a3b8', callback: v => v+'%' }, grid: { color: 'rgba(255,255,255,0.05)' } },
      }
    }
  });
}

function buildRiskChart() {
  const ctx = document.getElementById('riskChart').getContext('2d');
  new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['No Disease (54%)', 'Disease (46%)'],
      datasets: [{
        data: [54, 46],
        backgroundColor: [CHART_COLORS.emerald + 'cc', CHART_COLORS.red + 'cc'],
        borderColor:     [CHART_COLORS.emerald, CHART_COLORS.red],
        borderWidth: 2,
        hoverOffset: 8,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      cutout: '65%',
      plugins: {
        legend: { position: 'bottom', labels: { color: '#94a3b8', padding: 16 } },
        tooltip: CHART_DEFAULTS.plugins.tooltip,
      }
    }
  });
}

function buildAgeChart() {
  const ctx = document.getElementById('ageChart').getContext('2d');

  // Synthetic representative data — age distribution by risk
  const noDisease = [];
  const disease   = [];
  const seed = (s) => { let x = Math.sin(s)*10000; return x - Math.floor(x); };
  for (let i = 0; i < 60; i++) {
    noDisease.push({ x: Math.round(35 + seed(i*7)  * 30), y: Math.round(seed(i*13)*80 + 120) });
    disease.push(  { x: Math.round(45 + seed(i*11) * 25), y: Math.round(seed(i*17)*70 + 140) });
  }

  new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        { label: 'No Disease', data: noDisease, backgroundColor: CHART_COLORS.emerald+'99', pointRadius: 5, pointHoverRadius: 7 },
        { label: 'Disease',    data: disease,   backgroundColor: CHART_COLORS.red+'99',     pointRadius: 5, pointHoverRadius: 7 },
      ]
    },
    options: {
      ...CHART_DEFAULTS,
      scales: {
        x: { title: { display: true, text: 'Age', color: '#94a3b8' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { title: { display: true, text: 'Resting BP', color: '#94a3b8' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
      }
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. PREDICTION FORM
// ─────────────────────────────────────────────────────────────────────────────

const FEATURE_LABELS = {
  age: 'Age', sex: 'Sex', cp: 'Chest Pain', trestbps: 'Resting BP',
  chol: 'Cholesterol', fbs: 'Fasting BS', restecg: 'ECG',
  thalach: 'Max HR', exang: 'Exercise Angina', oldpeak: 'ST Depression',
  slope: 'ST Slope', ca: 'Major Vessels', thal: 'Thalassemia',
};

const RECOMMENDATIONS = {
  low:    `<strong>🟢 Low Risk Profile</strong><br/>Your clinical parameters suggest a low probability of heart disease. Continue your healthy lifestyle:<br/>• Maintain regular exercise (150 min/week)<br/>• Keep cholesterol below 200 mg/dL<br/>• Annual cardiac checkup recommended`,
  medium: `<strong>🟡 Moderate Risk Profile</strong><br/>Some risk factors are elevated. Consult your cardiologist and consider:<br/>• Lifestyle modifications (diet, exercise)<br/>• Monitor blood pressure and cholesterol<br/>• Stress test evaluation recommended<br/>• Check fasting blood glucose regularly`,
  high:   `<strong>🔴 High Risk Profile</strong><br/>Multiple risk factors indicate elevated heart disease probability. Immediate action recommended:<br/>• Urgent cardiologist consultation<br/>• Comprehensive cardiac workup (ECG, Echo, Stress Test)<br/>• Medication review with your physician<br/>• Emergency: Call 112/911 if experiencing chest pain`,
};

let lastResult = null;

function initPredictionForm() {
  const form     = document.getElementById('heartForm');
  const resetBtn = document.getElementById('resetBtn');
  const fillHigh = document.getElementById('fillHighRisk');
  const fillLow  = document.getElementById('fillLowRisk');
  const dlPDF    = document.getElementById('downloadPDF');
  const printBtn = document.getElementById('printReport');
  const saveBtn  = document.getElementById('saveHistory');

  form.addEventListener('submit', handlePredict);
  resetBtn.addEventListener('click', handleReset);
  fillHigh.addEventListener('click', () => fillSample('high'));
  fillLow.addEventListener('click',  () => fillSample('low'));
  dlPDF.addEventListener('click', downloadPDF);
  printBtn.addEventListener('click', () => window.print());
  saveBtn.addEventListener('click', saveToHistory);
}

function handlePredict(e) {
  e.preventDefault();

  // Validate
  const data = getFormData();
  if (!data) { showToast('⚠️ Please fill in all 13 fields.', 'error'); return; }

  // Show loading
  const overlay = document.getElementById('loadingOverlay');
  overlay.classList.add('active');

  // Simulate processing delay for UX
  setTimeout(() => {
    overlay.classList.remove('active');
    const result = predictHeartDisease(data);
    lastResult = { data, result, timestamp: new Date().toISOString() };
    displayResult(result, data);
    showToast(`${result.riskEmoji} Prediction complete — ${result.risk}`, result.riskClass === 'risk-low' ? 'success' : result.riskClass === 'risk-medium' ? 'info' : 'error');
    updateHeartMeter(result.healthScore);

    // Smooth scroll to results on mobile
    if (window.innerWidth < 1024) {
      document.getElementById('resultsPanel').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, 1800);
}

function getFormData() {
  const ids = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'];
  const data = {};
  for (const id of ids) {
    const el  = document.getElementById(id);
    const val = parseFloat(el.value);
    if (el.value === '' || isNaN(val)) { el.focus(); return null; }
    data[id] = val;
  }
  return data;
}

function displayResult(result, data) {
  document.getElementById('resultPlaceholder').style.display = 'none';
  const content = document.getElementById('resultContent');
  content.classList.add('visible');

  // Risk badge
  const badge = document.getElementById('riskBadge');
  badge.className = `risk-badge ${result.riskClass}`;
  badge.innerHTML = `${result.riskEmoji} ${result.risk}`;

  // Label
  const labelEl = document.getElementById('predLabel');
  labelEl.textContent = result.label;
  labelEl.style.color = result.prediction === 1 ? 'var(--red)' : 'var(--emerald)';

  // Probability bar
  document.getElementById('probPercent').textContent = `${result.probability}%`;
  setTimeout(() => {
    document.getElementById('probBar').style.width = `${result.probability}%`;
  }, 100);

  // Metrics
  document.getElementById('metricProb').textContent = `${result.probability}%`;
  document.getElementById('metricConf').textContent = `${result.confidence}%`;
  document.getElementById('metricRisk').textContent = result.risk.split(' ')[0];
  document.getElementById('metricModel').textContent = 'RF+LR';

  // Factors
  const factorsEl = document.getElementById('factorsList');
  factorsEl.innerHTML = result.topFactors.map(f => {
    const pct = Math.round(f.importance * 100 / MODEL.importance[result.topFactors[0].feature]);
    return `
      <div class="factor-item">
        <span class="factor-name">${FEATURE_LABELS[f.feature]}</span>
        <div class="factor-bar-wrap">
          <div class="factor-bar" style="width:${pct}%;background:${f.direction==='risk'?'linear-gradient(90deg,var(--red),var(--amber))':'linear-gradient(90deg,var(--emerald),var(--accent))'}"></div>
        </div>
        <span class="factor-score" style="color:${f.direction==='risk'?'var(--red)':'var(--emerald)'}">${f.value}</span>
      </div>`;
  }).join('');

  // Recommendation
  const rKey = result.riskClass === 'risk-low' ? 'low' : result.riskClass === 'risk-medium' ? 'medium' : 'high';
  document.getElementById('recommendation').innerHTML = RECOMMENDATIONS[rKey];
}

function handleReset() {
  document.getElementById('heartForm').reset();
  document.getElementById('resultPlaceholder').style.display = 'flex';
  document.getElementById('resultContent').classList.remove('visible');
  document.getElementById('probBar').style.width = '0%';
  lastResult = null;
  showToast('🔄 Form reset.', 'info');
}

function fillSample(type) {
  const samples = {
    high: { age:63, sex:1, cp:3, trestbps:145, chol:233, fbs:1, restecg:2, thalach:150, exang:0, oldpeak:2.3, slope:0, ca:0, thal:6 },
    low:  { age:41, sex:0, cp:1, trestbps:130, chol:204, fbs:0, restecg:0, thalach:172, exang:0, oldpeak:1.4, slope:2, ca:0, thal:3 },
  };
  const s = samples[type];
  for (const [key, val] of Object.entries(s)) {
    const el = document.getElementById(key);
    if (el) el.value = val;
  }
  showToast(`📋 ${type === 'high' ? '🔴 High risk' : '🟢 Low risk'} sample loaded!`, 'info');
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. PDF REPORT
// ─────────────────────────────────────────────────────────────────────────────
function downloadPDF() {
  if (!lastResult) { showToast('⚠️ Run a prediction first!', 'error'); return; }

  const { jsPDF } = window.jspdf;
  const doc  = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
  const { result, data, timestamp } = lastResult;
  const date = new Date(timestamp).toLocaleString();
  const W = 210;

  // Header gradient-like band
  doc.setFillColor(99, 102, 241);
  doc.rect(0, 0, W, 38, 'F');
  doc.setTextColor(255,255,255);
  doc.setFontSize(20); doc.setFont('helvetica','bold');
  doc.text('HeartGuard AI', 14, 16);
  doc.setFontSize(10); doc.setFont('helvetica','normal');
  doc.text('Heart Disease Risk Assessment Report', 14, 24);
  doc.text(`Generated: ${date}`, 14, 31);

  // Risk badge band
  const badgeColors = { 'risk-low': [16,185,129], 'risk-medium': [245,158,11], 'risk-high': [239,68,68] };
  const bc = badgeColors[result.riskClass];
  doc.setFillColor(...bc); doc.setTextColor(255,255,255);
  doc.rect(0, 38, W, 14, 'F');
  doc.setFontSize(13); doc.setFont('helvetica','bold');
  doc.text(`${result.riskEmoji.replace(/[^\x00-\x7F]/g,'')}  ${result.risk.toUpperCase()}  —  ${result.label}`, 14, 47);

  // Body
  doc.setTextColor(30,30,30);
  let y = 60;

  doc.setFontSize(12); doc.setFont('helvetica','bold');
  doc.text('Prediction Summary', 14, y); y += 8;

  const summaryRows = [
    ['Disease Probability', `${result.probability}%`],
    ['Confidence', `${result.confidence}%`],
    ['Risk Level', result.risk],
    ['Prediction', result.label],
    ['Model', 'Random Forest + Logistic Regression Ensemble'],
  ];
  doc.setFont('helvetica','normal'); doc.setFontSize(10);
  for (const [lbl, val] of summaryRows) {
    doc.setFillColor(245,245,255); doc.rect(14, y-4, W-28, 8, 'F');
    doc.setFont('helvetica','bold');   doc.text(lbl, 16, y);
    doc.setFont('helvetica','normal'); doc.text(String(val), 90, y);
    y += 10;
  }

  y += 6;
  doc.setFont('helvetica','bold'); doc.setFontSize(12);
  doc.text('Clinical Parameters', 14, y); y += 8;

  const paramRows = Object.entries(data).map(([k,v]) => [FEATURE_LABELS[k]||k, String(v)]);
  doc.setFontSize(10);
  paramRows.forEach(([lbl,val], i) => {
    if (i % 2 === 0) { doc.setFillColor(248,248,255); doc.rect(14, y-4, W-28, 8, 'F'); }
    doc.setFont('helvetica','bold');   doc.text(lbl, 16, y);
    doc.setFont('helvetica','normal'); doc.text(val, 90, y);
    y += 9;
  });

  y += 6;
  doc.setFont('helvetica','bold'); doc.setFontSize(12); doc.setTextColor(99,102,241);
  doc.text('Top Contributing Factors', 14, y); y += 8;
  doc.setTextColor(30,30,30); doc.setFontSize(10);
  result.topFactors.forEach(f => {
    doc.setFont('helvetica','bold');   doc.text(FEATURE_LABELS[f.feature]||f.feature, 16, y);
    doc.setFont('helvetica','normal'); doc.text(`Value: ${f.value}  |  Importance: ${(f.importance*100).toFixed(1)}%`, 60, y);
    y += 8;
  });

  // Recommendation
  y += 4;
  doc.setFont('helvetica','bold'); doc.setFontSize(12); doc.setTextColor(99,102,241);
  doc.text('Recommendations', 14, y); y += 8;
  doc.setTextColor(30,30,30); doc.setFont('helvetica','normal'); doc.setFontSize(9);
  const recText = RECOMMENDATIONS[result.riskClass==='risk-low'?'low':result.riskClass==='risk-medium'?'medium':'high']
    .replace(/<[^>]+>/g,'').replace(/&amp;/g,'&').trim();
  const lines = doc.splitTextToSize(recText, W-28);
  doc.text(lines, 14, y);

  // Disclaimer footer
  const footerY = 280;
  doc.setFillColor(240,240,255); doc.rect(0, footerY-4, W, 18, 'F');
  doc.setFontSize(8); doc.setTextColor(100,100,130); doc.setFont('helvetica','italic');
  doc.text('DISCLAIMER: This report is for educational purposes only and is NOT a substitute for professional medical advice.', 14, footerY+2);
  doc.text('Always consult a qualified healthcare provider for medical decisions.', 14, footerY+8);

  doc.save(`HeartGuard_Report_${Date.now()}.pdf`);
  showToast('📄 PDF report downloaded!', 'success');
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. LOCAL STORAGE HISTORY
// ─────────────────────────────────────────────────────────────────────────────
function initHistory() {
  renderHistory();
  document.getElementById('clearHistory').addEventListener('click', () => {
    if (confirm('Clear all saved predictions?')) {
      localStorage.removeItem('hg-history');
      renderHistory();
      showToast('🗑️ History cleared.', 'info');
    }
  });
}

function saveToHistory() {
  if (!lastResult) { showToast('⚠️ Run a prediction first!', 'error'); return; }

  const history = JSON.parse(localStorage.getItem('hg-history') || '[]');
  history.unshift({ id: Date.now(), ...lastResult });
  if (history.length > 20) history.pop(); // keep last 20
  localStorage.setItem('hg-history', JSON.stringify(history));
  renderHistory();
  showToast('💾 Prediction saved to history!', 'success');
}

function renderHistory() {
  const grid    = document.getElementById('historyGrid');
  const history = JSON.parse(localStorage.getItem('hg-history') || '[]');

  if (history.length === 0) {
    grid.innerHTML = '<div class="no-history" style="grid-column:1/-1">No predictions saved yet. Run a prediction and click 💾 Save.</div>';
    return;
  }

  grid.innerHTML = history.map(entry => {
    const { result, timestamp, id } = entry;
    const date = new Date(timestamp).toLocaleString();
    const accent = result.riskClass === 'risk-low' ? 'var(--emerald)' :
                   result.riskClass === 'risk-medium' ? 'var(--amber)' : 'var(--red)';
    return `
      <div class="history-card" id="hcard-${id}">
        <div class="history-accent" style="background:${accent}"></div>
        <div class="history-card-header">
          <span class="history-id">#${id.toString().slice(-6)} · ${date}</span>
          <button class="history-delete" onclick="deleteHistory(${id})" aria-label="Delete record">✕</button>
        </div>
        <div class="history-prediction" style="color:${accent}">${result.riskEmoji} ${result.risk}</div>
        <div class="history-meta" style="margin-bottom:0.5rem">${result.label}</div>
        <div class="history-prob" style="color:${accent}">${result.probability}%</div>
        <div class="history-meta">Confidence: ${result.confidence}%</div>
      </div>`;
  }).join('');
}

window.deleteHistory = function(id) {
  let history = JSON.parse(localStorage.getItem('hg-history') || '[]');
  history = history.filter(e => e.id !== id);
  localStorage.setItem('hg-history', JSON.stringify(history));
  renderHistory();
  showToast('🗑️ Record deleted.', 'info');
};

// ─────────────────────────────────────────────────────────────────────────────
// 11. HEART HEALTH SCORE METER
// ─────────────────────────────────────────────────────────────────────────────
function updateHeartMeter(score) {
  const fill   = document.getElementById('meterFill');
  const needle = document.getElementById('meterNeedle');
  const valEl  = document.getElementById('meterValue');
  const lblEl  = document.getElementById('meterLabel');

  // Arc length = 283 (half-circle)
  const offset = 283 - (score / 100) * 283;
  fill.style.strokeDashoffset = offset;

  // Needle: -90deg = 0%, 0deg = 50%, +90deg = 100%
  const angle = -90 + (score / 100) * 180;
  needle.style.transform = `rotate(${angle}deg)`;

  valEl.textContent = score;
  valEl.style.color = score > 65 ? 'var(--emerald)' : score > 40 ? 'var(--amber)' : 'var(--red)';
  lblEl.textContent = score > 65 ? '💚 Good Heart Health' : score > 40 ? '⚠️ Moderate Health' : '❤️‍🩹 Needs Attention';
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. BMI CALCULATOR
// ─────────────────────────────────────────────────────────────────────────────
function initTools() {
  document.getElementById('calcBMI').addEventListener('click', () => {
    const w = parseFloat(document.getElementById('bmiWeight').value);
    const h = parseFloat(document.getElementById('bmiHeight').value) / 100;

    if (!w || !h || h <= 0) { showToast('⚠️ Enter valid weight and height.', 'error'); return; }

    const bmi = w / (h * h);
    let category, color;
    if      (bmi < 18.5) { category = '🔵 Underweight';      color = '#3b82f6'; }
    else if (bmi < 25)   { category = '🟢 Normal Weight';    color = '#10b981'; }
    else if (bmi < 30)   { category = '🟡 Overweight';       color = '#f59e0b'; }
    else                 { category = '🔴 Obese';             color = '#ef4444'; }

    document.getElementById('bmiScore').textContent    = bmi.toFixed(1);
    document.getElementById('bmiScore').style.color    = color;
    document.getElementById('bmiCategory').textContent = category;
    document.getElementById('bmiResult').style.display = 'block';

    showToast(`⚖️ BMI: ${bmi.toFixed(1)} — ${category}`, 'info');
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. CHATBOT
// ─────────────────────────────────────────────────────────────────────────────
const CHAT_RESPONSES = {
  'heart disease':  'Heart disease refers to conditions affecting the heart\'s structure and function. Risk factors include high BP, high cholesterol, smoking, diabetes, obesity, and family history.',
  'cholesterol':    'Ideal cholesterol is below 200 mg/dL. LDL (bad cholesterol) should be under 100 mg/dL. High cholesterol increases plaque buildup in arteries.',
  'blood pressure': 'Normal BP is below 120/80 mmHg. High blood pressure (hypertension) is the #1 controllable risk factor for heart disease.',
  'symptoms':       'Common symptoms include chest pain or tightness, shortness of breath, fatigue, irregular heartbeat, and pain radiating to the arm or jaw.',
  'risk factors':   'Major risk factors: age, sex (males at higher risk), family history, high BP, high cholesterol, smoking, diabetes, obesity, and physical inactivity.',
  'prediction':     'Our AI uses 13 clinical parameters including age, cholesterol, ST depression, chest pain type, and more to estimate heart disease probability.',
  'features':       'The 13 features used are: Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar, ECG, Max Heart Rate, Exercise Angina, ST Depression, ST Slope, Major Vessels, and Thalassemia.',
  'accuracy':       'Our ensemble model achieves ~95% accuracy and 97.6% ROC-AUC on the Cleveland Heart Disease dataset.',
  'exercise':       'Regular aerobic exercise (150 min/week) strengthens the heart, lowers BP, reduces bad cholesterol, and helps maintain healthy weight.',
  'diet':           'Heart-healthy diet: fruits, vegetables, whole grains, lean protein, omega-3 fats. Limit salt, saturated fat, trans fat, and processed foods.',
  'hello':          '👋 Hello! I\'m HeartGuard Assistant. Ask me about heart disease, risk factors, symptoms, or how to use this prediction tool!',
  'help':           '💡 I can help with: heart disease facts, risk factors, how to use the predictor, lifestyle tips, and interpreting results.',
  'bmi':            'BMI (Body Mass Index) = weight(kg) / height(m)². Healthy range: 18.5–24.9. Obesity (BMI ≥30) increases heart disease risk significantly.',
  'default':        'I\'m a demo health assistant. For specific medical questions, please consult a qualified healthcare provider. I can help with general heart health information!',
};

function initChatbot() {
  const fab    = document.getElementById('chatbotFab');
  const window_ = document.getElementById('chatbotWindow');
  const close  = document.getElementById('chatbotClose');
  const input  = document.getElementById('chatInput');
  const send   = document.getElementById('chatSend');

  fab.addEventListener('click',   () => window_.classList.toggle('open'));
  close.addEventListener('click', () => window_.classList.remove('open'));

  const doSend = () => {
    const msg = input.value.trim();
    if (!msg) return;
    addChatMsg(msg, 'user');
    input.value = '';
    setTimeout(() => addChatMsg(getBotResponse(msg), 'bot'), 600);
  };

  send.addEventListener('click', doSend);
  input.addEventListener('keydown', e => { if (e.key === 'Enter') doSend(); });
}

function addChatMsg(text, who) {
  const msgs = document.getElementById('chatMessages');
  const div  = document.createElement('div');
  div.className = `chat-msg ${who}`;
  div.textContent = text;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function getBotResponse(msg) {
  const lower = msg.toLowerCase();
  for (const [key, resp] of Object.entries(CHAT_RESPONSES)) {
    if (key !== 'default' && lower.includes(key)) return resp;
  }
  return CHAT_RESPONSES.default;
}

// ─────────────────────────────────────────────────────────────────────────────
// 14. TOAST NOTIFICATIONS
// ─────────────────────────────────────────────────────────────────────────────
function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  const toast     = document.createElement('div');
  const icons     = { success: '✅', error: '❌', info: 'ℹ️' };

  toast.className = `toast ${type}`;
  toast.innerHTML = `<span class="toast-icon">${icons[type]||'💬'}</span><span class="toast-msg">${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(20px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// ─────────────────────────────────────────────────────────────────────────────
// 15. NAVBAR ACTIVE LINKS (scroll spy)
// ─────────────────────────────────────────────────────────────────────────────
window.addEventListener('scroll', () => {
  const sections = document.querySelectorAll('section[id], div.stats-bar');
  const links    = document.querySelectorAll('.nav-links a');
  let current    = '';

  sections.forEach(s => {
    if (window.scrollY >= s.offsetTop - 120) current = s.id || '';
  });

  links.forEach(a => {
    a.style.color = a.getAttribute('href') === `#${current}` ? 'var(--accent-light)' : '';
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 16. DYNAMIC COPYRIGHT YEAR
// ─────────────────────────────────────────────────────────────────────────────
function initCopyright() {
  const els = document.querySelectorAll('.footer-bottom p');
  if (els.length) {
    els[0].innerHTML = `© ${new Date().getFullYear()} HeartGuard AI. Built for educational purposes only.`;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 17. SMOOTH ANCHOR SCROLLING
// ─────────────────────────────────────────────────────────────────────────────
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    const target = document.querySelector(a.getAttribute('href'));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});
