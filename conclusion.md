# 💡 Key Findings

## 🚧 Main Challenge
The dataset was heavily imbalanced:

- Severity Class 2 accounted for ~86.5% of all records.
- Initial models achieved ~87% accuracy by predicting only Class 2.
- Minority classes (1, 3, 4) had near-zero precision and recall.

This highlighted why accuracy alone is misleading for imbalanced datasets.

---

## 🛠️ Techniques Used

- Applied **SMOTE** to balance classes for XGBoost.
- Used **Class Weights** for Logistic Regression and ANN models.
- Evaluated models using **F1-score, Precision, and Recall** instead of accuracy.

![Actual vs Predicted Severity](visualisations/actual_vs_predicted.png)

---

## 🔑 Important Insights

- `temperature_f` showed a positive correlation with accident severity.
- `visibility_mi` showed a negative correlation with severity.
- Logistic Regression provided the best interpretability among tested models.

![Logistic Regression Coefficients](visualisations/log_reg_coefs.png)

---

## 🚀 Future Improvements

Potential features to improve prediction quality:

- Weather conditions (rain, fog, snow)
- Time-based traffic patterns
- Road type and speed limits
- Additional traffic and environmental data
