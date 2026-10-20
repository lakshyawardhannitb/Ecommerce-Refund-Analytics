"""
Module 1: Data Collection, Loading, and Quality Checking
AI-Powered E-Commerce Return & Refund Risk Analytics
"""

import pandas as pd
import numpy as np
import os

DATA_PATH = "ecommerce_returns_synthetic_data.csv"
REPORT_PATH = "outputs/reports/data_quality_report.txt"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the dataset from CSV."""
    df = pd.read_csv(path)
    print(f"[LOAD] Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def data_quality_check(df: pd.DataFrame) -> dict:
    """Perform a comprehensive data quality audit."""
    report = {}

    # --- Basic Info ---
    report["shape"] = df.shape
    report["columns"] = list(df.columns)
    report["dtypes"] = df.dtypes.astype(str).to_dict()

    # --- Missing Values ---
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report["missing_counts"] = missing[missing > 0].to_dict()
    report["missing_pct"] = missing_pct[missing_pct > 0].to_dict()

    # --- Duplicates ---
    report["duplicate_rows"] = int(df.duplicated().sum())

    # --- Unique values per column ---
    report["unique_values"] = {col: int(df[col].nunique()) for col in df.columns}

    # --- Numeric summary ---
    report["numeric_summary"] = df.describe().round(2).to_dict()

    # --- Value counts for categoricals ---
    cat_cols = ["Return_Status", "Product_Category", "Payment_Method",
                "Shipping_Method", "Return_Reason", "User_Gender"]
    report["value_counts"] = {}
    for col in cat_cols:
        if col in df.columns:
            report["value_counts"][col] = df[col].value_counts(dropna=False).to_dict()

    # --- Outlier detection (IQR method) ---
    num_cols = ["Product_Price", "Order_Quantity", "Days_to_Return",
                "User_Age", "Discount_Applied"]
    report["outliers"] = {}
    for col in num_cols:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
            report["outliers"][col] = {
                "lower_fence": round(lower, 2),
                "upper_fence": round(upper, 2),
                "outlier_count": n_out,
            }

    return report


def print_quality_report(report: dict) -> str:
    """Pretty-print and save the quality report."""
    lines = []
    lines.append("=" * 70)
    lines.append("  DATA QUALITY REPORT -- E-Commerce Returns Dataset")
    lines.append("=" * 70)

    r, c = report["shape"]
    lines.append(f"\n? Shape            : {r:,} rows x {c} columns")
    lines.append(f"? Duplicate rows   : {report['duplicate_rows']}")

    lines.append("\n-- Column Data Types --")
    for col, dtype in report["dtypes"].items():
        lines.append(f"  {col:<25} {dtype}")

    if report["missing_counts"]:
        lines.append("\n-- Missing Values --")
        for col, cnt in report["missing_counts"].items():
            pct = report["missing_pct"].get(col, 0)
            lines.append(f"  {col:<25} {cnt:>6,}  ({pct:.1f}%)")
    else:
        lines.append("\n[OK] No missing values detected.")

    lines.append("\n-- Unique Values Per Column --")
    for col, n in report["unique_values"].items():
        lines.append(f"  {col:<25} {n:>6,}")

    lines.append("\n-- Categorical Value Counts --")
    for col, vc in report["value_counts"].items():
        lines.append(f"\n  [{col}]")
        for val, cnt in vc.items():
            lines.append(f"    {str(val):<30} {cnt:>6,}")

    lines.append("\n-- Outlier Detection (IQR Method) --")
    for col, info in report["outliers"].items():
        lines.append(
            f"  {col:<25} outliers={info['outlier_count']:>5}  "
            f"fence=[{info['lower_fence']}, {info['upper_fence']}]"
        )

    lines.append("\n-- Numeric Summary --")
    numeric_df = pd.DataFrame(report["numeric_summary"]).round(2)
    lines.append(numeric_df.to_string())

    lines.append("\n" + "=" * 70)
    text = "\n".join(lines)
    print(text)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n[SAVED] Quality report -> {REPORT_PATH}")
    return text


if __name__ == "__main__":
    df = load_data()
    report = data_quality_check(df)
    print_quality_report(report)
