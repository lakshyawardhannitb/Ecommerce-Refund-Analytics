# AI-Powered E-Commerce Return & Refund Risk Analytics

> An end-to-end machine learning pipeline that ingests e-commerce order data, performs exploratory analytics, engineers features, trains and tunes multiple classifiers, explains predictions with SHAP, and surfaces actionable business recommendations — all deployable through a Streamlit dashboard or a CLI prediction app.

---

## Project Structure

```
.
├── main.py                          # Full pipeline orchestrator (run this first)
├── app.py                           # Streamlit web dashboard
├── ecommerce_returns_synthetic_data.csv
├── requirements.txt
├── src/
│   ├── s01_data_loading_and_quality.py
│   ├── s02_data_cleaning_and_analytics.py
│   ├── s03_eda_and_visualization.py
│   ├── s04_feature_engineering.py
│   ├── s05_ml_pipeline.py
│   ├── s06_explainable_ai_and_insights.py
│   └── s07_prediction_app.py
└── outputs/
    ├── models/                      # Trained model (.pkl) + feature_names.txt
    ├── plots/                       # 21 saved visualizations
    └── reports/                     # Text reports (quality, analytics, ML, insights)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python main.py
```

This executes all 10 pipeline steps end-to-end and saves all outputs under `outputs/`.

### 3. Launch the Streamlit dashboard

```bash
streamlit run app.py
```

### 4. Use the CLI prediction app

```bash
# Interactive single-order prediction
python src/s07_prediction_app.py

# Batch prediction from CSV
python src/s07_prediction_app.py --batch path/to/new_orders.csv
```

---

## Dataset

| Property | Value |
|---|---|
| File | `ecommerce_returns_synthetic_data.csv` |
| link  https://www.kaggle.com/datasets/sayalikhot21/synthetic-dataset-for-e-commerce-return-analysis?select=ecommerce_returns_synthetic_data.csv 

| Rows | 10,000 orders |
| Columns | 17 |
| Date range | 2023 – 2024 (731 unique order dates) |
| Target | `Return_Status` → `Is_Returned` (binary: 0/1) |

**Key columns:** `Order_ID`, `Product_Category`, `Product_Price`, `Order_Quantity`, `Discount_Applied`, `Payment_Method`, `Shipping_Method`, `User_Age`, `User_Gender`, `User_Location`, `Return_Status`, `Return_Reason`, `Days_to_Return`

---

## Key Results

| Metric | Value |
|---|---|
| Total Orders | 10,000 |
| Overall Return Rate | **50.52%** |
| Total Gross Revenue | $7,626,183.56 |
| Total Net Revenue | $5,713,232.49 |
| Revenue At Risk | **$2,871,592.10** |
| Best ML Model | Random Forest (Tuned) |
| Top SHAP Feature | Order Day-of-Week |

### Model Comparison (Test Set)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| K-Nearest Neighbors | 50.85% | 51.33% | 51.49% | 51.41% | 50.23% |
| Logistic Regression | 50.85% | 51.17% | 58.32% | 54.51% | 49.95% |
| Decision Tree | 49.75% | 50.25% | 49.11% | 49.67% | 49.76% |
| AdaBoost | 50.75% | 50.96% | 65.84% | 57.45% | 49.54% |
| Random Forest | 49.50% | 50.00% | 50.59% | 50.30% | 49.50% |
| Naive Bayes | 49.80% | 50.26% | 56.44% | 53.17% | 49.23% |
| Gradient Boosting | 49.65% | 50.13% | 55.84% | 52.83% | 48.99% |
| SVM | 48.40% | 48.99% | 53.07% | 50.95% | 48.01% |

> All models achieved near-random performance (~50% ROC-AUC), which is expected given the dataset is synthetically generated with no strong signal between features and the return label.

---

## Pipeline Steps

| Step | Module | Description |
|---|---|---|
| 01 | `s01` | Data loading and quality audit |
| 02 | `s01` | Missing value, outlier & duplicate checks |
| 03 | `s02` | Data cleaning — date parsing, imputation, type corrections |
| 04 | `s02` | Sales calculation — Gross, Net, Revenue At Risk |
| 05 | `s02` | Descriptive analytics & grouping/aggregation |
| 06 | `s03` | EDA — 14 visualizations |
| 07 | `s04` | Feature engineering (11 new features, 23 total) |
| 08 | `s05` | ML — 8 model comparison + 5-fold CV |
| 09 | `s05` | Hyperparameter tuning (GridSearchCV — RF & GB) |
| 10 | `s06` | Explainable AI — SHAP beeswarm, bar, waterfall, dependence |
| 11 | `s06` | AI-assisted business insights & recommendations |

---

## Technologies

- **Python 3.x**, Pandas, NumPy
- **Scikit-learn** — 8 classifiers, GridSearchCV, StratifiedKFold
- **SHAP** — TreeExplainer, beeswarm / bar / waterfall / dependence plots
- **Matplotlib, Seaborn** — 21 saved visualisations
- **Streamlit** — interactive web dashboard
- **Joblib** — model persistence

---

## Outputs

| Path | Contents |
|---|---|
| `outputs/plots/01–14_*.png` | EDA visualizations |
| `outputs/plots/15_model_comparison.png` | Multi-model bar chart |
| `outputs/plots/16_confusion_roc_*.png` | Confusion matrix + ROC curve (tuned models) |
| `outputs/plots/17_feature_importance_*.png` | Feature importances (tuned models) |
| `outputs/plots/18–21_shap_*.png` | SHAP summary, bar, waterfall, dependence |
| `outputs/models/random_forest_(tuned).pkl` | Saved best model |
| `outputs/models/feature_names.txt` | 23 feature names |
| `outputs/reports/data_quality_report.txt` | Full data audit |
| `outputs/reports/descriptive_analytics_report.txt` | KPIs & grouped stats |
| `outputs/reports/ml_model_comparison.txt` | All-model metric table |
| `outputs/reports/business_insights_report.txt` | AI recommendations |

---

## License

This project was developed as part of an internship. All data is synthetically generated and contains no real customer information.
