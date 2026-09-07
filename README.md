# 🚗 Car Price Prediction

A data analytics and machine learning project for analyzing automobile prices and predicting car prices using Python, Pandas, SQLite, Linear Regression, and Streamlit.

## 📌 Project Overview

This project analyzes automobile data and provides an interactive Streamlit dashboard.

The project includes:

- Automobile data cleaning and preprocessing
- Data analysis using Pandas
- SQLite database for storing cleaned data
- Automobile filtering by brand and production year
- Average price analysis by brand
- Car price prediction using Linear Regression
- Interactive Streamlit dashboard

## 🛠️ Technologies

- Python
- Pandas
- SQLite
- Scikit-learn
- Linear Regression
- Plotly
- Streamlit
- Jupyter Notebook

## 📊 Features

1. Filters

Users can filter automobiles using advanced options:
- Brand (Marka)
- City (Şəhər)
- Production year range (Buraxılış ili)

2. Automobile Table

The dashboard displays a comprehensive table with all dataset columns for the automobiles matching the selected filters:
- Ad ID (Elan_id)
- Brand & Model (Marka, Model)
- Production Year & Car Age (Buraxılış_ili, Yaş)
- Engine Size (Mühərrik_Həcmi) & Mileage (Yürüş_KM)
- Fuel Type & Transmission (Yanacaq, Sürətlər_Qutusu)
- Body Type & Color (Ban_novu, Rəng)
- City (Şəhər)
- Accident & Paint Status (Vuruq, Rənglənib)
- Number of Owners (Sahib_sayi)
- Ad Date & Price (Elan_Tarixi, Qiymət_AZN)
- Price Category (Qiymət_Kateqoriyası)


3. Interactive Visual Analytics

Interactive Plotly charts provide deep insights into the filtered market data:
- Scatter plot showing Automobile Age vs Price, colored by Accident Status.
- Bar chart illustrating Average Price and Total Listing Count grouped by Color.

4. Machine Learning Price Prediction

Users can input a complete set of vehicle specifications to receive a real-time price estimation powered by a trained **Random Forest Regression** model:
- Brand & Model selection
- Production Year (Automobile Age is automatically calculated)
- Engine Size & Mileage
- Fuel Type & Transmission
- Body Type, Color, & City
- Number of Owners
- Accident & Paint condition status

## 📁 Project Structure

```text
car-price-prediction/
│
├── app.py
├── avtomobil_elanlari.csv
├── seherler.csv
├── project.ipynb
├── avtomobil.db
├── requirements.txt
├── README.md
└── .gitignore

## 👨‍💻 Author

Bəhmən Sarıyev

Information Technologies @ ADA University
