# Precision Cane Analytics: Yield & Quality Forecasting 🎋
[![Icesi University](https://img.shields.io/badge/University-Icesi-004a87.svg)](https://www.icesi.edu.co/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![ABET Accredited](https://img.shields.io/badge/Program-ABET%20Accredited-orange.svg)](https://www.abet.org/)
[![ML Framework](https://img.shields.io/badge/Framework-Scikit--Learn%20%2F%20XGBoost-green.svg)](https://scikit-learn.org/)

## Executive Summary
This repository hosts a high-performance machine learning pipeline developed for **Ingenio Providencia** Cali, Colombia. The project addresses critical industrial needs for precision agriculture by transforming historical crop data into actionable insights for harvest optimization.

**Analytical Objectives:**
1.  **Yield Prediction (Regression):** Forecasting *Toneladas de Caña por Hectárea* (TCH) to optimize mill logistics.
2.  **Quality Tiering (Classification):** Categorizing sucrose levels into **High, Medium, and Low** tiers to maximize industrial extraction efficiency.

---

## Engineering Standards & Methodology
 We strictly adhere to the **CRISP-DM** methodology, integrating engineering principles to solve complex problems.

### Data Governance & Ethics 
* **Confidentiality:** In compliance with institutional ethics, raw datasets are git-ignored and excluded from public access.
* **Agronomic Imputation:** Implementation of domain-specific strategies (e.g., median imputation for outliers and constant `0` for non-applied ripening agents).
* **Multicollinearity Control:** Rigorous Variance Inflation Factor (VIF) analysis to ensure model interpretability.

### Team Responsibilities & Workflow
The project is modularized to reflect a real-world engineering team structure:
* **01_EDA & Data Cleansing:** Isabella / Luis
* **02_Linear_Regression (Baseline):** Isabella
* **03_Logistic_Regression_&_KNN:** Melisa
* **04_Random_Forest_Ensembles:** Luis
* **05_XGBoost_&_Final_Comparison:** Santiago

---

## Repository Structure
```text
├── data/
│  ├── raw/           # Original confidential datasets
│  └── processed/     # Cleaned dataframes with engineered targets
├── notebooks/     
│  ├── 01_EDA.ipynb    # Exploratory Analysis & Feature Engineering
│  ├── 02_Linear_Reg.ipynb # Baseline Regression (TCH & Sucrose)
│  ├── 03_Class_Base.ipynb # Logistic & KNN Benchmarks
│  ├── 04_RandomForest.ipynb # Advanced Ensemble Modeling
│  └── 05_XGBoost_Eval.ipynb # Final Optimization & Model Comparison
├── src/          # Modularized Production Code
│  ├── preprocessing.py  # Imputation & Tier Generation logic
│  └── evaluation.py    # Visual reports as Confusion Matrix, F1, Kappa
├── reports/        # IEEE Format Technical Documentation
└── README.md
```

## Tech Stack
Core: Python 3.11, Scikit-Learn, XGBoost.

Analytics: Pandas, NumPy, Scipy (VIF analysis).

Visualization: Seaborn, Matplotlib.

## Strategic Business Insights
Predictive Power: Identification of "Crop Age" and "Ripening Dosage" as the most influential features for sucrose accumulation.

Operational Impact: By tiering lot performance into tertiles, the mill can prioritize harvesting schedules based on real-time quality estimates, reducing industrial loss.


Cali, Colombia - 2026
Transforming industrial data into strategic competitive advantages through Scalable AI.