# Precision Cane Analytics: Yield & Quality Forecasting 🎋

## 📌 Executive Summary
This repository hosts   high-performance machine learning pipeline developed for **Ingenio Providencia** (Cali, Colombia). The project addresses the industrial need for precision agriculture by transforming historical crop data into actionable insights for harvest optimization.

We implement dual-objective analytical approach:
1.  **Yield Prediction (Regression):** Forecasting Toneladas de Caña por Hectárea (TCH) to synchronize field productivity with mill logistics.
2.  **Quality Tiering (Classification):** Categorizing sucrose levels (%Sac.Caña) into High, Medium, and Low tiers to optimize the industrial extraction process.

---

## 🏗️ Methodology & Engineering Standards
The project strictly follows the **CRISP-DM** methodology, ensuring that every technical decision is rooted in business requirements.

### 1. Data Governance & Ethics (SO4)
* **Data Privacy:** In accordance with institutional policies, raw datasets (`HISTORICO_SUERTES.xlsx` and `BD_IPSA_1940.xlsx`) are strictly confidential and excluded from this public repository.
* **Leakage Mitigation:** Rigorous feature selection to remove post-harvest indicators ( .g., TAHM, ATR), ensuring the model is   true predictive tool and not   descriptive one.
* **Agronomic Imputation:** Domain-specific strategies for missing values, such as imputing ripening dosages based on crop cycles rather than simple statistical averages.

### 2. Predictive Pipeline
We utilize   "Benchmark-to-Advanced" strategy to validate performance improvements:
* **Baseline Models:** Multiple Linear Regression (verified via VIF for multicollinearity) and K-Nearest Neighbors (KNN).
* **Advanced Ensembles:** Implementation of **Random Forest** and **XGBoost** to capture complex, non-linear interactions between soil type, variety, and climate.
* **Validation Rigor:** Employment of **10-Fold Cross-Validation** and Hold-out (80/20) strategies to guarantee model generalizability.

---

## 🛠️ Tech Stack
| Category | Tools |
| :--- | :--- |
| **Language** | Python 3.x |
| **Data Science** | Pandas, NumPy, Scipy (VIF analysis) |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Visualization** | Seaborn, Matplotlib|

---

## 📂 Repository Structure
```text
├── data/               # (Git-ignored) Confidential industry datasets
├── notebooks/          
│   ├── 1.0-EDA-Regression-Analysis.ipynb    # Data exploration & Baseline models
│   └── 2.0-Advanced-Classification.ipynb    # Ensemble methods & Hyperparameter tuning
├── src/                
│   ├── preprocessing.py # Feature engineering & cleaning scripts
│   └── evaluation.py    # Custom metrics (RMSE, Kappa, F1-Score)
├── reports/            # Technical documentation & Business summary
└── README.md
```

---

Strategic Business InsightsPredictive Power: Identification of key drivers such as Crop Age and Ripening Dosage as critical predictors for high-sucrose yields.

Operational Impact: By classifying lot performance using tertiles, the mill can prioritize harvesting schedules based on real-time quality estimates .

---

Cali, Colombia - 2026

Transforming raw industrial data into strategic competitive advantages through Scalable AI.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![ABET Accredited](https://img.shields.io/badge/Program-ABET%20Accredited-orange.svg)](https://www.abet.org/)
[![ML Framework](https://img.shields.io/badge/Framework-Scikit--Learn%20%2F%20XGBoost-green.svg)](https://scikit-learn.org/)