# 🚦 Traffic Accident Severity Prediction: Tackling Class Imbalance

This repository contains a machine learning pipeline designed to predict the severity of traffic accidents based on environmental conditions. The project explores the challenges of **severe class imbalance** in real-world accident data and demonstrates how to build models that correctly identify minority classes.

---

## 🎯 Objective

The objective is to move beyond the deceptive metric of overall "accuracy" when dealing with highly skewed datasets. By comparing a baseline model against implementations using **SMOTE** and **Class Weights**, this project aims to accurately predict severe and minor accidents that are otherwise masked by the overwhelming majority of standard accidents.

---

## 🛠️ Tech Stack

* **Python 3.x**
* **Pandas & NumPy** (Data manipulation)
* **Scikit-Learn** (Logistic Regression, scaling, evaluation)
* **XGBoost** (Gradient-boosted decision trees)
* **TensorFlow & Keras** (Artificial Neural Networks)
* **Imbalanced-learn** (SMOTE resampling)

---

## 📂 Dataset

The analysis utilizes a subset of traffic accident data, focusing specifically on South Carolina (SC) to test the models on regional environmental factors.
* **Target Variable:** `severity` (Scale 1-4)
* **Key Features:** `temperature_f` and `visibility_mi`
* **Data File:** `data/sc_accidents.csv` 

---

## 📈 Analysis & Visualizations

### 1. Feature Importance and Model Coefficients
This visualization breaks down the internal logic of the balanced Logistic Regression model, highlighting how temperature and visibility influence accident severity.

![Logistic Regression Coefficients](images/log_reg_coefs.png)

### 2. Actual vs. Predicted Severity
This plot tracks the true accident severity against the model's predictions over a sample set, proving that the resampling techniques successfully mapped the minority classes.

![Actual vs Predicted Severity](images/actual_vs_predicted.png)

---

## 📝 Conclusion Summary

Relying on overall accuracy is a flawed approach for this dataset. While baseline models achieved ~87% accuracy, they failed to identify minority classes. Implementing SMOTE and adjusting class weights forced the models to learn the actual underlying patterns. 

*For a detailed breakdown of the findings, please see [`conclusion.md`](conclusion.md).*

---

## 🚀 How to Run this Project

1. **Clone the repository:**
    git clone https://github.com/suchismitab0511/accident_pred

2. **Install dependencies:**
    pip install -r requirements.txt

3. **Run the notebook:** Open `accident_pred.ipynb` in Jupyter Notebook or Google Colab and run all cells.