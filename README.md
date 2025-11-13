# Customer_Churn_Prediction-using-ML
Data Analyst &amp; ML Enthusiast | Python • Pandas • Power BI • SQL • Machine Learning | Exploring data-driven decision making and predictive analytics
# 📘 Customer Churn Prediction using Machine Learning

### 👨‍💻 Author: [Ayush Kumar Srivas](https://github.com/AyushKumarSrivas)
### 🗂️ Repository: Customer_Churn_Prediction

---

## 🧩 Overview

Customer churn prediction is one of the most crucial challenges for businesses, especially in the telecom and subscription industries.  
This project aims to **analyze customer behavior** and **predict churn likelihood** using **Machine Learning models**.  

The project combines **Exploratory Data Analysis (EDA)** for insight discovery and **predictive modeling** to build data-driven retention strategies.

---

## 🧠 Objectives

- Perform **EDA** to understand factors influencing churn.  
- Handle data imbalance using **SMOTE**.  
- Train and compare multiple **machine learning models**.  
- Derive **business insights** and **recommendations** from the results.  
- Save the best model using **pickle** for future predictions.

---

## 📊 Dataset Description

The dataset contains customer-related information such as:

| Feature | Description |
|----------|-------------|
| `gender` | Gender of the customer |
| `SeniorCitizen` | Indicates if the customer is a senior citizen |
| `Partner` | Whether the customer has a partner |
| `Dependents` | Whether the customer has dependents |
| `tenure` | Number of months the customer has stayed |
| `PhoneService` | Whether the customer has a phone service |
| `MultipleLines` | Availability of multiple lines |
| `InternetService` | Type of internet service |
| `Contract` | Contract type (Month-to-month, One year, Two year) |
| `PaymentMethod` | Method used for payment |
| `MonthlyCharges` | Monthly payment amount |
| `TotalCharges` | Total amount charged to the customer |
| `Churn` | Target column (Yes = Churned, No = Retained) |

---

## 🧰 Libraries Used

- **Python**
- **NumPy**, **Pandas** → Data manipulation
- **Matplotlib**, **Seaborn** → Data visualization (box plots, heatmaps)
- **Scikit-learn** → Preprocessing, Model training & Evaluation
- **Imbalanced-learn (SMOTE)** → Oversampling minority class
- **XGBoost** → Gradient boosting model
- **Pickle** → Model saving

---

## 📈 Exploratory Data Analysis (EDA)

EDA was performed to uncover key patterns and trends using **visualization and statistical analysis**.

### 🔍 Key EDA Steps
- Checked data types, missing values, and unique counts.  
- Visualized the distribution of categorical and numerical variables.  
- Used **box plots** to analyze how `MonthlyCharges` and `TotalCharges` differ between churned and retained customers.  
- Plotted a **correlation heatmap** to identify relationships between numerical variables.  
- Analyzed churn distribution across various demographics and service types.

### 🧠 Insights from EDA
- Customers with **higher MonthlyCharges and lower TotalCharges** are more likely to churn.  
- Customers on **month-to-month contracts** show a much higher churn rate compared to those on yearly plans.  
- **Electronic check** payment method customers churn more often.  
- Long-term contracts significantly reduce churn risk.

---

## ⚙️ Data Preprocessing Steps

1. **Handling Missing Values**  
   Checked and filled or removed null values appropriately.

2. **Encoding Categorical Variables**  
   Used `LabelEncoder` to convert categorical columns into numeric form.

3. **Balancing Classes**  
   Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to fix class imbalance.

4. **Train-Test Split**  
   Split dataset into 80% training and 20% testing data.

---

## 🤖 Model Building

Trained multiple classification models for churn prediction:

| Model | Description |
|--------|-------------|
| **Decision Tree Classifier** | Simple tree-based classifier |
| **Random Forest Classifier** | Ensemble of decision trees for better accuracy |
| **XGBoost Classifier** | Gradient boosting model, optimized for performance |

---

## 🧪 Model Evaluation

Evaluated models using accuracy, confusion matrix, and classification report.

| Model | Accuracy | Remarks |
|--------|-----------|----------|
| Decision Tree | 85% | Decent, but prone to overfitting |
| Random Forest | 88% | Good performance and stability |
| XGBoost | 90% | Best accuracy and generalization |

*(Replace with your actual accuracy values if different)*

---

### 📉 Confusion Matrix Example (Random Forest)
```python
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d')
plt.title("Confusion Matrix - Random Forest")
plt.show()
