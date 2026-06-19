# Telco Customer Churn Prediction

A machine learning project that analyzes the IBM Telco Customer Churn dataset to predict whether a customer will churn (cancel their subscription). The notebook covers the full data science pipeline — from exploratory analysis through model evaluation.

## Dataset

IBM Sample Dataset — publicly available via [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

The dataset contains customer information for a fictional telecom company, including demographics, account details, subscribed services, and a binary churn label.

**Key features:**

| Category | Columns |
|----------|---------|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Target | `Churn` (Yes / No) |

## Project Structure

```
.
├── Untitled.ipynb                        # Main analysis notebook

```

## Workflow

### 1. Data Loading & Exploration
- Load the CSV with pandas and inspect the first few rows, shape, and data types.
- Identify categorical vs. numerical columns.
- Check for missing values and class distribution of the target variable.

### 2. Exploratory Data Analysis (EDA)
- Bar charts of all categorical feature distributions (normalized).
- Correlation heatmap for numerical features.
- Visual comparison of churn vs. non-churn populations across key features.

### 3. Preprocessing
- **Label Encoding** applied to all 18 categorical columns (including the target `Churn`) using `sklearn.preprocessing.LabelEncoder`.
- The `customerID` column is dropped as it carries no predictive signal.
- Features (`X`) and target (`Y`) are separated, then split into training and test sets.

### 4. Models Trained

| Model | Library |
|-------|---------|
| Support Vector Classifier (SVC) | `sklearn.svm` |
| Random Forest Classifier | `sklearn.ensemble` |
| Gradient Boosting Classifier | `sklearn.ensemble` |

### 5. Evaluation
Each model is evaluated using:
- **Accuracy score**
- **Classification report** (precision, recall, F1-score)
- **Confusion matrix**
- **ROC curve & AUC score** (plotted for Gradient Boosting)

## Requirements

```
python >= 3.9
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Clone or download the repository.
2. Place the dataset in the same directory as the notebook.
3. Open the notebook and run all cells top to bottom:

```bash
jupyter notebook Untitled.ipynb
```

## Results

Three classifiers were trained and compared. The **Gradient Boosting Classifier** produces an ROC curve with its AUC displayed in the final plot, serving as the primary benchmark for model performance. Refer to the classification reports inside the notebook for per-class precision, recall, and F1 scores.


