"""
main.py -- Full Pipeline Orchestrator
AI-Powered E-Commerce Return & Refund Risk Analytics and Prediction System

Runs all 17 project components end-to-end:
  ? Data Loading & Quality Check
  ? Data Cleaning
  ? Sales Calculation
  ? Descriptive Analytics
  ? EDA & Data Visualization (14 plots)
  ? Feature Engineering
  ? ML Model Comparison (8 models)
  ? Cross-Validation
  ? Hyperparameter Tuning (RF + GB)
  ? Final Model Evaluation (confusion matrix, ROC curve, feature importance)
  ? Explainable AI -- SHAP analysis (beeswarm, bar, waterfall, dependence)
  ? Business Insights & AI Recommendations
  ? Save model artifacts
  ? Prediction Application ready

Usage:
    python main.py
"""

import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")

# -- Pipeline Imports ----------------------------------------------
from src.s01_data_loading_and_quality    import load_data, data_quality_check, print_quality_report
from src.s02_data_cleaning_and_analytics import clean_data, calculate_sales, descriptive_analytics, print_analytics_report
from src.s03_eda_and_visualization       import run_all_eda_plots
from src.s04_feature_engineering         import engineer_features, prepare_ml_features
from src.s05_ml_pipeline                 import run_ml_pipeline
from src.s06_explainable_ai_and_insights import run_shap_analysis, generate_business_insights


# -----------------------------------------------------------------
BANNER = """
+======================================================================+
|                                                                      |
|   AI-Powered E-Commerce Return & Refund Risk Analytics               |
|   and Prediction System                                              |
|                                                                      |
|   Technologies: Python | Pandas | NumPy | Matplotlib | Seaborn      |
|                 Scikit-learn | SHAP | Joblib                         |
|                                                                      |
+======================================================================+
"""


def section(title: str, step: int):
    print(f"\n{'='*70}")
    print(f"  STEP {step:02d} -- {title}")
    print(f"{'='*70}\n")


def main():
    t0 = time.time()
    print(BANNER)

    # ----------------------------------------------
    # STEP 01 -- Data Loading & Quality Check
    # ----------------------------------------------
    section("Data Collection & Loading", 1)
    df_raw = load_data("ecommerce_returns_synthetic_data.csv")

    section("Data Quality Checking", 2)
    quality_report = data_quality_check(df_raw)
    print_quality_report(quality_report)

    # ----------------------------------------------
    # STEP 02 -- Data Cleaning
    # ----------------------------------------------
    section("Data Cleaning", 3)
    df_clean = clean_data(df_raw)

    # ----------------------------------------------
    # STEP 03 -- Sales Calculation
    # ----------------------------------------------
    section("Sales Calculation", 4)
    df_sales = calculate_sales(df_clean)

    # ----------------------------------------------
    # STEP 04 -- Descriptive Analytics & Grouping
    # ----------------------------------------------
    section("Descriptive Analytics & Grouping / Aggregation", 5)
    analytics_results = descriptive_analytics(df_sales)
    print_analytics_report(analytics_results)

    # ----------------------------------------------
    # STEP 05 -- EDA & Visualizations
    # ----------------------------------------------
    section("Exploratory Data Analysis & Data Visualization (14 plots)", 6)
    run_all_eda_plots(df_sales)

    # ----------------------------------------------
    # STEP 06 -- Feature Engineering
    # ----------------------------------------------
    section("Feature Engineering", 7)
    df_fe = engineer_features(df_sales)
    X, y, feature_names = prepare_ml_features(df_fe)

    # ----------------------------------------------
    # STEP 07-09 -- ML: Comparison, CV, Tuning
    # ----------------------------------------------
    section("Machine Learning -- Model Comparison, CV & Hyperparameter Tuning", 8)
    ml_results = run_ml_pipeline(X, y, feature_names)

    best_model = ml_results["best_model"]
    X_train    = ml_results["X_train"]
    X_test     = ml_results["X_test"]
    y_test     = ml_results["y_test"]

    # ----------------------------------------------
    # STEP 10 -- Explainable AI (SHAP)
    # ----------------------------------------------
    section("Explainable AI -- SHAP Analysis", 9)
    shap_values, explainer = run_shap_analysis(
        best_model, X_train, X_test,
        feature_names, ml_results["best_name"]
    )

    # ----------------------------------------------
    # STEP 11 -- Business Insights & Recommendations
    # ----------------------------------------------
    section("Business Insights & AI-Assisted Recommendations", 10)
    generate_business_insights(df_fe, shap_values, feature_names)

    # ----------------------------------------------
    # DONE
    # ----------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE  --  {elapsed:.1f} seconds")
    print(f"{'='*70}")
    print("\n  Outputs saved to:")
    print("    outputs/plots/        -- All EDA, ML & SHAP visualizations")
    print("    outputs/models/       -- Trained model & feature names")
    print("    outputs/reports/      -- Quality, analytics & insights reports")
    print("\n  To launch the prediction app, run:")
    print("    python src/s07_prediction_app.py\n")


if __name__ == "__main__":
    main()
