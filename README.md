# 🚗 Vehicle Price Prediction Web App

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2-purple?style=for-the-badge&logo=pandas)](https://pandas.pydata.org/)

This project is a machine learning-powered web application designed to predict the market price of vehicles. It leverages a robust data cleaning and feature engineering pipeline to train a Random Forest Regressor, which is then deployed using an interactive web interface built with Streamlit.

---
## 🚀 Live Demo

You can try out the live prediction model at the following link:

**[➡️ Live Website Link](https://thunderer9506-vehicle-prediction-website-home-bszcq9.streamlit.app/Prediction)**

---
## ✨ Features

* **Interactive UI:** A user-friendly interface for selecting various vehicle attributes.
* **Dynamic & Dependent Dropdowns:** The list of available models updates automatically based on the selected make.
* **Advanced Feature Engineering:** Extracts valuable information from complex text fields like `engine` and `transmission` to improve accuracy.
* **Real-time Prediction:** Instantly predicts the vehicle's estimated price based on the user's selections.

---
## 🛠️ Technologies Used

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Web Framework:** Streamlit
* **Data Visualization (in Notebook):** Matplotlib, Seaborn

---
## 🧠 The Machine Learning Pipeline

The model's high accuracy (R² score of **0.82**) is a result of a meticulous data processing and feature engineering pipeline.

1.  **Data Cleaning:** Missing values were handled using context-aware imputation. For example, missing `fuel` types for electric vehicles were correctly identified and filled.
2.  **Outlier Handling:** Statistical outlier detection using the Interquartile Range (IQR) method was applied to numerical columns like `price` and `mileage` to create a more robust model.
3.  **Feature Engineering:** This was a critical step to extract maximum value from the data:
    * **Text Feature Extraction:** Numerical and binary features like `engine_liters`, `gears`, `is_turbo`, and `is_cvt` were extracted from complex text columns.
    * **High Cardinality Management:** Categorical features with too many unique values (like `model` and `trim`) were cleaned by grouping rare instances into an "Other" category.
4.  **Encoding:** **Label Encoding** was used to convert all categorical features into a numerical format suitable for the Random Forest model.
5.  **Modeling:** A **Random Forest Regressor** was trained and fine-tuned using a systematic hyperparameter search to achieve the best possible performance.

---
## 📂 Project Structure
.
/Pages <br>
  |--Prediction.py<br>
├── 📄 cleanedData.csv<br>
├── 📄 Hello.py<br>
├── 📄 README.md<br>
├── 📄 requirements.txt<br>
└── 📓 Vehicle Prediction.ipynb<br>
