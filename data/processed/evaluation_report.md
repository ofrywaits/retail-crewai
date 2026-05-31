# Evaluation Report — Israeli Retail Profit Classifier

**תאריך:** 2026-05-31 17:27
**מטרה:** חיזוי האם עסקה תניב רווח גבוה (מעל החציון)
**גודל Test Set:** 1,980 עסקאות

## השוואת מודלים

| מודל | Accuracy | F1 Score | AUC |
|------|---------|---------|-----|
| Random Forest | 0.8939 | 0.8933 | 0.9613 |
| Logistic Regression | 0.752 | 0.7517 | 0.7916 |

**מודל מומלץ: Random Forest**
Random Forest עדיף ב-F1: 0.1416


## Random Forest

| מדד | ערך |
|-----|-----|
| Accuracy | 0.8939 |
| F1 Score (weighted) | 0.8933 |

### Classification Report
```
              precision    recall  f1-score   support

  Low Profit       0.97      0.82      0.89       995
 High Profit       0.84      0.97      0.90       985

    accuracy                           0.89      1980
   macro avg       0.90      0.89      0.89      1980
weighted avg       0.90      0.89      0.89      1980

```

### Confusion Matrix
|  | Predicted Low | Predicted High |
|--|--------------|----------------|
| **Actual Low**  | 813 | 182 |
| **Actual High** | 28 | 957 |

### Feature Importance (Top 5)
| Feature | Importance |
|---------|-----------|
| קטגוריה_enc | 0.5476 |
| מחיר_יחידה | 0.2307 |
| כמות | 0.0392 |
| month | 0.0364 |
| רשת_enc | 0.0352 |


## Logistic Regression

| מדד | ערך |
|-----|-----|
| Accuracy | 0.7520 |
| F1 Score (weighted) | 0.7517 |

### Classification Report
```
              precision    recall  f1-score   support

  Low Profit       0.77      0.72      0.74       995
 High Profit       0.73      0.79      0.76       985

    accuracy                           0.75      1980
   macro avg       0.75      0.75      0.75      1980
weighted avg       0.75      0.75      0.75      1980

```

### Confusion Matrix
|  | Predicted Low | Predicted High |
|--|--------------|----------------|
| **Actual Low**  | 713 | 282 |
| **Actual High** | 209 | 776 |


---
*נוצר אוטומטית על ידי Evaluation Agent — CrewAI Israeli Retail Project*
