# AI-Based Diamond Price Prediction and Analytics System

> **Project Disclaimer:** This project is built for educational and research purposes. Its predictions are estimates from a public dataset and must not be treated as professional gemological appraisal or a guaranteed market price.

---

## Project Overview

This project implements an end-to-end machine learning workflow for estimating diamond prices from the classic 4 Cs and physical measurements. It covers data quality analysis, exploratory analysis, cleaning, feature engineering, model comparison, hyperparameter tuning, and evaluation through three ordered Jupyter notebooks.

---

## Problem Statement

Given a diamond's carat weight, quality grades, and physical dimensions, predict its retail price in US dollars using supervised regression.

---

## Objectives

1. Inspect the dataset and identify invalid or unusual records.
2. Clean physically impossible measurements without leaking target information.
3. Explore relationships between diamond quality, dimensions, and price.
4. Engineer useful geometric features such as volume and density.
5. Train and compare multiple regression algorithms.
6. Tune a strong baseline with `GridSearchCV`.
7. Document model performance and practical limitations.

---

## Dataset

- **Source:** [ggplot2 diamonds dataset](https://github.com/tidyverse/ggplot2/blob/main/data-raw/diamonds.csv)
- **Records:** 53,940 raw rows; 53,920 rows after cleaning
- **Features:** 9 inputs and 1 numeric target
- **Target:** `price`, measured in US dollars
- **Price range:** $326 to $18,823

### Features

| Column | Type | Description |
|---|---|---|
| `carat` | Numeric | Diamond mass in carats |
| `cut` | Ordinal | Fair, Good, Very Good, Premium, or Ideal |
| `color` | Ordinal | J (lowest grade) through D (highest grade) |
| `clarity` | Ordinal | I1 through IF clarity grades |
| `depth` | Numeric | Total depth percentage |
| `table` | Numeric | Top width relative to the widest point |
| `x`, `y`, `z` | Numeric | Length, width, and depth in millimetres |
| `price` | Numeric target | Retail price in US dollars |

---

## Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.9+ |
| Data processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Machine learning | scikit-learn |
| Interactive analysis | Jupyter Notebook |

---

## Project Structure

```text
DiamondPricePrediction/
├── data/
│   ├── diamonds.csv              # Raw dataset
│   └── diamonds_cleaned.csv      # Cleaned and feature-engineered data
├── model/
│   ├── diamond_price_pipeline.pkl # Generated serialized model
│   └── model_meta.json             # Generated model metadata
├── frontend/
│   ├── streamlit_app.py            # Streamlit prediction interface
│   └── README.md
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   ├── data_cleaning.ipynb
│   └── model_training.ipynb
├── src/
│   └── diamond_price_prediction/
│       ├── preprocessing.py        # Loading, cleaning, and features
│       └── evaluation.py           # Metrics and evaluation plots
├── outputs/
│   ├── figures/                   # Generated charts
│   ├── metrics/                   # Model comparison and importance CSVs
│   └── predictions/               # Held-out predictions
├── report/
│   ├── README.md
│   └── Diamond_Price_Project_Report.docx # Generated report
├── train_model.py                 # Reproducible training and artifact generation
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## How to Run

Run the notebooks in this order from the project root:

```bash
jupyter notebook notebooks/exploratory_data_analysis.ipynb
jupyter notebook notebooks/data_cleaning.ipynb
jupyter notebook notebooks/model_training.ipynb
```

The cleaning notebook creates `data/diamonds_cleaned.csv`, which is used by the model-building notebook.

### Train the deployment model

```bash
python train_model.py
```

This trains the comparison models, serializes `model/diamond_price_pipeline.pkl`, and generates figures, metrics, predictions, and the project report.

### Run the Streamlit frontend

```bash
streamlit run frontend/streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

---

## Model Evaluation Results

Seven regression models were compared. Metrics below are from the project notebook's held-out evaluation.

| Model | R² | RMSE ($) | MAE ($) | MAPE |
|---|---:|---:|---:|---:|
| Gradient Boosting | 0.9827 | 527 | 271 | 0.075 |
| Random Forest | 0.9820 | 537 | 260 | 0.064 |
| Decision Tree | 0.9753 | 629 | 318 | 0.086 |
| KNN (K=11) | 0.9719 | 672 | 362 | 0.101 |
| Lasso | 0.9121 | 1,187 | 803 | 0.437 |
| Ridge | 0.7058 | 2,172 | 798 | 0.434 |
| Linear Regression | 0.6881 | 2,236 | 800 | 0.435 |
| **Random Forest (Tuned)** | **0.9820** | **537** | **263** | **0.066** |

The tuned Random Forest used `n_estimators=200`, `max_depth=15`, and `min_samples_leaf=1`, with cross-validation R² of 0.9804. Gradient Boosting achieved the highest untuned R² in the comparison table.

---

## Key Findings

- Carat is the strongest individual price signal and is closely related to the physical dimensions.
- Tree-based ensembles capture the non-linear relationship between size, quality, and price more effectively than the raw linear models.
- The quality grades become more informative when comparing diamonds with similar carat weights.
- A log transformation of `price` is a promising next step for improving linear-model behavior.

---

## Data Cleaning Summary

| Issue | Resolution |
|---|---|
| Zero values in `x`, `y`, or `z` | Removed 20 physically impossible records |
| Categorical quality columns | Encoded using domain-specific ordinal orderings |
| Derived geometry | Added `volume`, `density`, and `log_carat` features |
| Missing derived values | Filled numeric missing values with medians |
| Duplicate rows | Checked during notebook analysis |

---

## Limitations

- The dataset contains historical prices and may not represent current market prices.
- Predictions are estimates, not certified appraisal values.
- Tree-model performance may vary on diamonds outside the dataset's feature distribution.
- The project does not include uncertainty intervals, external validation, or live market data.
- Ordinal encoding simplifies complex gemological quality relationships.

---

## Future Improvements

- Add a log-target modeling pipeline and compare it using the same evaluation split.
- Use cross-validated hyperparameter search for all candidate models.
- Add prediction intervals and explainability with permutation importance or SHAP.
- Build a small Streamlit interface for interactive price estimation.
- Validate against a newer and more diverse market dataset.

---

## References

1. [ggplot2 diamonds dataset](https://github.com/tidyverse/ggplot2/blob/main/data-raw/diamonds.csv)
2. [Scikit-learn documentation](https://scikit-learn.org/)
3. [Jupyter documentation](https://docs.jupyter.org/)
