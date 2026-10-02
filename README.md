📱 Mobile Price Prediction

A Machine Learning based web application that predicts the price range of a mobile phone using its hardware and feature specifications.

🚀 Project Overview

This project uses a Random Forest Classifier to classify mobile phones into four price-range categories based on specifications such as RAM, battery power, internal memory, camera, screen resolution, connectivity and other features.

🧠 Machine Learning Model

- Algorithm: Random Forest Classifier
- Dataset: Mobile Price Classification
- Training Data: 80% (1600 samples)
- Testing Data: 20% (400 samples)
- Accuracy: 88.00%
- Number of Classes: 4

📊 Price Range Classes

- Class 0 — Low
- Class 1 — Medium
- Class 2 — High
- Class 3 — Very High

🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit

🌐 Web Application

The project includes a Streamlit-based web interface where users can enter mobile specifications and get a predicted price category.

📁 Project Structure

Mobile-price-prediction/
├── app.py
├── main.py
├── dataset/
│   └── train.csv
└── models/
    └── mobile_price_model.pkl

🎯 Objective

The objective of this project is to demonstrate how Machine Learning can be used to classify mobile phones into different price-range categories based on their specifications.
