---
title: Used Car Price Prediction
emoji: 🏎️
colorFrom: blue
colorTo: green
sdk: streamlit

python_version: "3.10"
app_file: app.py
pinned: false
license: gpl-3.0
---

# Regression of Used Car Prices 🚗

This project is part of the **Kaggle Playground Series (S4E9)**. It aims to develop a high-accuracy regression model to predict the resale price of used cars by analyzing technical specifications and historical data[cite: 1].

## 🎯 Aim & Target
The primary goal is to minimize **RMSE** and handle high-cardinality categorical data. The target variable is `price`, representing the continuous numerical listing price of a vehicle[cite: 1].

## 🛠️ Tech Stack
- **Python:** 3.10 (Optimized for Intel-based hardware)
- **Framework:** Streamlit
- **Model:** Gradient Boosting / XGBoost
- **Libraries:** Scikit-learn, Pandas, Re (Regex for feature extraction)

## 📊 Feature Engineering
The application extracts critical signals from messy string data, including:
- **Engine Specs:** Horsepower, Liter, and Forced Induction via Regex.
- **Color Analysis:** Identification of "Premium Paint" (Metallic/Pearl finishes).
- **Vehicle Age:** Calculated from the `model_year`[cite: 1].

## 🚀 How to Run Locally
1. Clone the repository.
2. Create a virtual environment: `python3.10 -m venv venv`.
3. Activate it: `source venv/bin/activate`.
4. Install requirements: `pip install -r requirements.txt`.
5. Run the app: `streamlit run app.py`.