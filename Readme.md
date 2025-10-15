# 🏦 Loan Default Prediction - End-to-End Data Science Project

[![R](https://img.shields.io/badge/R-276DC3?style=flat&logo=r&logoColor=white)](https://www.r-project.org/)
[![tidymodels](https://img.shields.io/badge/tidymodels-latest-blue)](https://www.tidymodels.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A complete machine learning pipeline built in R for predicting loan approval status using classification algorithms. This project demonstrates end-to-end data science workflow from data generation to model deployment.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models](#models)
- [Results](#results)
- [Visualizations](#visualizations)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project predicts whether a loan application will be **approved (Y)** or **rejected (N)** based on applicant characteristics. It implements two machine learning models:

- **Logistic Regression** - Baseline interpretable model
- **Random Forest** - Advanced ensemble method

The project follows best practices including proper train-test splitting, feature engineering, data preprocessing, and comprehensive model evaluation.

---

## ✨ Features

- **Automated Data Generation**: Creates synthetic loan application dataset
- **Data Cleaning Pipeline**: Handles missing values, duplicates, and data types
- **Feature Engineering**: Creates derived features like loan-to-income ratio
- **Multiple Models**: Implements and compares Logistic Regression and Random Forest
- **Comprehensive EDA**: Generates correlation plots, distributions, and relationship visualizations
- **Model Evaluation**: ROC-AUC, accuracy, confusion matrices
- **Feature Importance**: Identifies key predictors of loan approval
- **Reproducible**: Uses `set.seed()` for consistent results
- **Well-Organized**: Structured folder hierarchy for data, outputs, and models

---

## 📁 Project Structure

```
loan-default-prediction/
│
├── data/                          # Data directory
│   ├── loan_data.csv             # Raw dataset
│   ├── train_data.csv            # Training set (80%)
│   └── test_data.csv             # Testing set (20%)
│
├── outputs/                       # Generated outputs
│   ├── correlation_plot.png
│   ├── loan_status_distribution.png
│   ├── loan_amount_vs_income.png
│   ├── income_distribution_by_status.png
│   ├── default_rate_by_credit_history.png
│   ├── feature_importance.csv
│   ├── feature_importance_plot.png
│   └── random_forest_loan_model.rds
│
├── scripts/                       # R scripts
│   └── loan_prediction.R         # Main analysis script
│
├── models/                        # Saved models
│
├── reports/                       # Analysis reports
│
└── README.md                      # Project documentation
```

---

## 🚀 Installation

### Prerequisites

- R (≥ 4.0.0)
- RStudio (recommended)

### Required R Packages

```r
install.packages(c(
  "tidyverse",
  "janitor",
  "skimr",
  "tidymodels",
  "themis",
  "vip",
  "pROC",
  "corrplot",
  "rsample",
  "recipes",
  "parsnip",
  "workflows",
  "yardstick",
  "ranger"
))
```

### Clone Repository

```bash
git clone https://github.com/yourusername/loan-default-prediction.git
cd loan-default-prediction
```

---

## 📊 Dataset

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `Gender` | Categorical | Male/Female |
| `Married` | Categorical | Yes/No |
| `ApplicantIncome` | Numeric | Monthly income in dollars |
| `LoanAmount` | Numeric | Requested loan amount (in thousands) |
| `Credit_History` | Binary | 1 = Good credit, 0 = Poor credit |
| `Loan_Status` | Binary | Y = Approved, N = Rejected (Target) |

### Engineered Features

- **Loan_to_Income**: Ratio of loan amount to applicant income (risk indicator)

### Dataset Statistics

- **Total Records**: 500 applications
- **Training Set**: 400 records (80%)
- **Testing Set**: 100 records (20%)
- **Class Distribution**: ~70% Approved, ~30% Rejected

---

## 🔬 Methodology

### 1. Data Generation
Synthetic dataset created with realistic distributions to simulate loan application data.

### 2. Data Cleaning
- Remove duplicate records
- Handle missing values using median/mode imputation
- Convert categorical variables to factors
- Binary encoding of target variable

### 3. Feature Engineering
- Create `loan_to_income` ratio
- Normalize numeric features
- One-hot encode categorical variables

### 4. Train-Test Split
Stratified 80-20 split ensuring balanced class distribution in both sets.

### 5. Preprocessing Pipeline
Using `tidymodels` recipes:
- Dummy variable encoding for categorical features
- Normalization of numeric predictors

### 6. Model Training
- **Logistic Regression**: GLM engine
- **Random Forest**: Ranger engine with 500 trees, mtry=3, min_n=5

### 7. Evaluation
- Accuracy, Kappa metrics
- ROC-AUC scores
- Confusion matrices
- Feature importance analysis

---

## 🤖 Models

### Logistic Regression
```r
logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")
```

**Strengths**: Interpretable, fast, works well with linear relationships

### Random Forest
```r
rand_forest(mtry = 3, trees = 500, min_n = 5) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")
```

**Strengths**: Handles non-linear relationships, robust to outliers, high accuracy

---

## 📈 Results

### Model Performance

| Model | Accuracy | ROC-AUC | Strengths |
|-------|----------|---------|-----------|
| Logistic Regression | ~XX% | ~0.XX | Interpretable, baseline |
| Random Forest | ~XX% | ~0.XX | Higher accuracy, ensemble |

*Note: Actual metrics will vary due to random data generation*

### Key Insights

The feature importance analysis reveals which factors most strongly influence loan approval decisions. Typically:
- Credit history has the strongest impact
- Loan-to-income ratio is a critical risk indicator
- Applicant income influences approval likelihood

---

## 📊 Visualizations

The project generates several analytical visualizations:

1. **Correlation Plot**: Relationships between numeric features
2. **Loan Status Distribution**: Class balance visualization
3. **Loan Amount vs Income Scatter**: Decision boundary insights
4. **Income Distribution by Status**: Overlapping histograms
5. **Default Rate by Credit History**: Bar chart showing risk patterns
6. **Feature Importance**: Ranked predictor significance

All plots are saved in the `outputs/` directory as PNG files.

---

## 💻 Usage

### Run Complete Pipeline

```r
source("scripts/loan_prediction.R")
```

### Load Trained Model

```r
model <- readRDS("outputs/random_forest_loan_model.rds")

# Make predictions on new data
new_data <- data.frame(
  gender = "Male",
  married = "Yes",
  applicant_income = 5000,
  loan_amount = 150,
  credit_history = 1,
  loan_to_income = 150/5000
)

predictions <- predict(model, new_data)
```

### Generate Custom Predictions

```r
# Load test data
test_data <- read_csv("data/test_data.csv")

# Predict probabilities
predictions <- predict(model, test_data, type = "prob")
```

---

## 📦 Dependencies

### Core Packages
- `tidyverse` (v2.0.0+): Data manipulation and visualization
- `tidymodels` (v1.0.0+): Machine learning framework

### Modeling
- `ranger`: Random Forest implementation
- `parsnip`: Unified modeling interface
- `recipes`: Feature engineering
- `workflows`: Model workflow management

### Evaluation
- `yardstick`: Model metrics
- `pROC`: ROC curve analysis
- `vip`: Variable importance

### Data Processing
- `janitor`: Data cleaning
- `skimr`: Data summarization
- `rsample`: Data splitting

### Visualization
- `corrplot`: Correlation matrices
- `ggplot2`: Graphics (included in tidyverse)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Aayush Chalke** - *Initial work* - [Chalke-Aayush-15](https://github.com/Chalke-Aayush-15/Finance-Loan-Default-Prediction)

---

## 🙏 Acknowledgments

- Built using the `tidymodels` ecosystem
- Inspired by real-world financial risk assessment challenges
- Thanks to the R community for excellent documentation and packages

---

## 📧 Contact

For questions or feedback, please reach out:
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Name](https://linkedin.com/in/yourprofile)

---

## 🔮 Future Enhancements

- [ ] Add cross-validation for more robust evaluation
- [ ] Implement XGBoost and compare performance
- [ ] Create interactive Shiny dashboard
- [ ] Add hyperparameter tuning with grid search
- [ ] Incorporate SMOTE for handling class imbalance
- [ ] Deploy model as REST API using plumber
- [ ] Add time-series analysis for temporal patterns
- [ ] Create automated reporting with RMarkdown

---

**⭐ If you find this project helpful, please give it a star!**