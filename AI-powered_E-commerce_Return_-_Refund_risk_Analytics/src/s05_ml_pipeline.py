"""
Module 5: Machine Learning -- Model Training, Comparison & Hyperparameter Tuning
AI-Powered E-Commerce Return & Refund Risk Analytics
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

MODEL_DIR  = "outputs/models"
PLOT_DIR   = "outputs/plots"
REPORT_DIR = "outputs/reports"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

RANDOM_STATE = 42


# ---------------------------------------------
# 1. SPLIT DATA
# ---------------------------------------------

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[ML] Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------
# 2. MODEL COMPARISON
# ---------------------------------------------

def get_base_models() -> dict:
    """Return a dictionary of baseline classifiers."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7)),
        ]),
        "Naive Bayes": GaussianNB(),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=RANDOM_STATE)),
        ]),
    }


def evaluate_model(name: str, model, X_train, X_test, y_train, y_test) -> dict:
    """Train a model and return evaluation metrics."""
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Model": name,
        "Accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
        "Precision": round(precision_score(y_test, y_pred) * 100, 2),
        "Recall":    round(recall_score(y_test, y_pred) * 100, 2),
        "F1 Score":  round(f1_score(y_test, y_pred) * 100, 2),
        "ROC-AUC":   round(roc_auc_score(y_test, y_proba) * 100, 2) if y_proba is not None else "N/A",
    }
    return metrics, model


def compare_models(X_train, X_test, y_train, y_test) -> tuple:
    """Train and compare all baseline models. Returns comparison DataFrame and trained models."""
    print("\n[ML] Running model comparison...")
    models = get_base_models()
    results = []
    trained = {}

    for name, model in models.items():
        metrics, trained_model = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(metrics)
        trained[name] = trained_model
        print(f"  ok {name:<25} Acc={metrics['Accuracy']}%  F1={metrics['F1 Score']}%  AUC={metrics['ROC-AUC']}%")

    comparison_df = pd.DataFrame(results).set_index("Model").sort_values("ROC-AUC", ascending=False)
    print("\n[ML] Model Comparison (sorted by ROC-AUC):")
    print(comparison_df.to_string())
    return comparison_df, trained


def plot_model_comparison(comparison_df: pd.DataFrame):
    """Bar chart comparing all models across metrics."""
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    df_plot = comparison_df[metrics].reset_index()
    df_melt = df_plot.melt(id_vars="Model", var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=df_melt, x="Model", y="Score", hue="Metric",
                palette="Set2", ax=ax)
    ax.set_title("Model Comparison -- All Metrics (%)", fontweight="bold", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Score (%)")
    ax.legend(loc="lower right", fontsize=9)
    plt.xticks(rotation=25, ha="right")
    path = os.path.join(PLOT_DIR, "15_model_comparison.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [PLOT] Saved -> {path}")


# ---------------------------------------------
# 3. CROSS-VALIDATION
# ---------------------------------------------

def cross_validate_best(model, X_train, y_train, cv: int = 5) -> np.ndarray:
    """Run stratified k-fold CV and print results."""
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=cv_splitter,
                             scoring="roc_auc", n_jobs=-1)
    print(f"\n[ML] Cross-Validation ({cv}-Fold ROC-AUC):")
    print(f"  Scores  : {np.round(scores, 4)}")
    print(f"  Mean    : {scores.mean():.4f}  ?  {scores.std():.4f}")
    return scores


# ---------------------------------------------
# 4. HYPERPARAMETER TUNING
# ---------------------------------------------

def tune_random_forest(X_train, y_train) -> RandomForestClassifier:
    """GridSearch hyperparameter tuning for Random Forest."""
    print("\n[ML] Hyperparameter tuning -- Random Forest (GridSearchCV)...")

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
    }

    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    cv_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        rf, param_grid, cv=cv_splitter,
        scoring="roc_auc", n_jobs=-1, verbose=1, refit=True,
    )
    grid_search.fit(X_train, y_train)

    print(f"  Best Params : {grid_search.best_params_}")
    print(f"  Best AUC    : {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


def tune_gradient_boosting(X_train, y_train) -> GradientBoostingClassifier:
    """GridSearch hyperparameter tuning for Gradient Boosting."""
    print("\n[ML] Hyperparameter tuning -- Gradient Boosting (GridSearchCV)...")

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
    }

    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
    cv_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        gb, param_grid, cv=cv_splitter,
        scoring="roc_auc", n_jobs=-1, verbose=1, refit=True,
    )
    grid_search.fit(X_train, y_train)

    print(f"  Best Params : {grid_search.best_params_}")
    print(f"  Best AUC    : {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


# ---------------------------------------------
# 5. FINAL MODEL EVALUATION
# ---------------------------------------------

def full_evaluation(model, model_name: str, X_test, y_test, feature_names: list):
    """Detailed evaluation: report + confusion matrix + ROC curve."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n[ML] -- {model_name} -- Final Evaluation --")
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"  Precision : {precision_score(y_test, y_pred)*100:.2f}%")
    print(f"  Recall    : {recall_score(y_test, y_pred)*100:.2f}%")
    print(f"  F1 Score  : {f1_score(y_test, y_pred)*100:.2f}%")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, y_proba)*100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Returned", "Returned"]))

    # -- Confusion matrix --
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Not Returned", "Returned"]).plot(
        ax=axes[0], colorbar=False, cmap="Blues"
    )
    axes[0].set_title(f"Confusion Matrix -- {model_name}", fontweight="bold")

    # -- ROC Curve --
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)
    axes[1].plot(fpr, tpr, lw=2, color="#3b82d4",
                 label=f"ROC AUC = {auc_val:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title(f"ROC Curve -- {model_name}", fontweight="bold")
    axes[1].legend(loc="lower right")

    fname = model_name.lower().replace(" ", "_")
    path = os.path.join(PLOT_DIR, f"16_confusion_roc_{fname}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [PLOT] Saved -> {path}")

    # -- Feature importance --
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=feature_names)
        fi_top = fi.sort_values(ascending=False).head(15)

        fig2, ax2 = plt.subplots(figsize=(9, 6))
        fi_top.sort_values().plot(kind="barh", ax=ax2, color="#3b82d4")
        ax2.set_title(f"Top 15 Feature Importances -- {model_name}", fontweight="bold")
        ax2.set_xlabel("Importance")
        path2 = os.path.join(PLOT_DIR, f"17_feature_importance_{fname}.png")
        fig2.savefig(path2, bbox_inches="tight", dpi=150)
        plt.close(fig2)
        print(f"  [PLOT] Saved -> {path2}")


# ---------------------------------------------
# 6. SAVE MODEL
# ---------------------------------------------

def save_model(model, name: str, scaler=None):
    """Persist model (and optional scaler) to disk."""
    fname = name.lower().replace(" ", "_")
    path = os.path.join(MODEL_DIR, f"{fname}.pkl")
    joblib.dump(model, path)
    print(f"[ML] Model saved -> {path}")
    if scaler:
        s_path = os.path.join(MODEL_DIR, "scaler.pkl")
        joblib.dump(scaler, s_path)
        print(f"[ML] Scaler saved -> {s_path}")
    return path


# ---------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------

def run_ml_pipeline(X: pd.DataFrame, y: pd.Series, feature_names: list) -> dict:
    """
    Full ML pipeline:
      1. Split
      2. Compare 8 baseline models
      3. CV on top-2 models
      4. Hyperparameter tune top-2
      5. Final evaluation
      6. Save best model
    Returns dict with train/test splits, best model, and comparison table.
    """
    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Compare baseline models
    comparison_df, trained_models = compare_models(X_train, X_test, y_train, y_test)
    plot_model_comparison(comparison_df)

    # Save comparison report
    report_path = os.path.join(REPORT_DIR, "ml_model_comparison.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("MODEL COMPARISON REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(comparison_df.to_string())
    print(f"[ML] Comparison report -> {report_path}")

    # Cross-validate top 2 models
    top_models = comparison_df.index[:2].tolist()
    for name in top_models:
        cross_validate_best(trained_models[name], X_train, y_train)

    # Hyperparameter tuning
    print("\n[ML] Starting hyperparameter tuning for top 2 models...")
    tuned_rf = tune_random_forest(X_train, y_train)
    tuned_gb = tune_gradient_boosting(X_train, y_train)

    # Final evaluation of both tuned models
    full_evaluation(tuned_rf, "Random Forest (Tuned)", X_test, y_test, feature_names)
    full_evaluation(tuned_gb, "Gradient Boosting (Tuned)", X_test, y_test, feature_names)

    # Pick best based on AUC
    auc_rf = roc_auc_score(y_test, tuned_rf.predict_proba(X_test)[:, 1])
    auc_gb = roc_auc_score(y_test, tuned_gb.predict_proba(X_test)[:, 1])

    if auc_rf >= auc_gb:
        best_model = tuned_rf
        best_name  = "Random Forest (Tuned)"
    else:
        best_model = tuned_gb
        best_name  = "Gradient Boosting (Tuned)"

    print(f"\n[ML] ? Best model selected: {best_name} (AUC={max(auc_rf, auc_gb):.4f})")
    save_model(best_model, best_name)

    # Save feature names
    feat_path = os.path.join(MODEL_DIR, "feature_names.txt")
    with open(feat_path, "w") as f:
        f.write("\n".join(feature_names))
    print(f"[ML] Feature names saved -> {feat_path}")

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "comparison": comparison_df,
        "best_model": best_model,
        "best_name": best_name,
        "tuned_rf": tuned_rf,
        "tuned_gb": tuned_gb,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.s01_data_loading_and_quality import load_data
    from src.s02_data_cleaning_and_analytics import clean_data, calculate_sales
    from src.s04_feature_engineering import engineer_features, prepare_ml_features

    df = calculate_sales(clean_data(load_data()))
    df_fe = engineer_features(df)
    X, y, feat_names = prepare_ml_features(df_fe)
    results = run_ml_pipeline(X, y, feat_names)
