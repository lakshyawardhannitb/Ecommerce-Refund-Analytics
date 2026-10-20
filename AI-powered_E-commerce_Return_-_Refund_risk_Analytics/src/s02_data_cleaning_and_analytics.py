"""
Module 2: Data Cleaning, Sales Calculation & Descriptive Analytics
AI-Powered E-Commerce Return & Refund Risk Analytics
"""

import pandas as pd
import numpy as np
import os


REPORT_PATH = "outputs/reports/descriptive_analytics_report.txt"


# ---------------------------------------------
# DATA CLEANING
# ---------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform all data cleaning steps:
      - Parse dates
      - Fill / flag missing values
      - Standardize categoricals
      - Remove duplicates
      - Correct data types
    Returns a clean copy of the dataframe.
    """
    df = df.copy()
    original_rows = len(df)

    # 1. Drop full-duplicate rows
    df.drop_duplicates(inplace=True)
    removed = original_rows - len(df)
    if removed:
        print(f"[CLEAN] Removed {removed} duplicate rows.")

    # 2. Parse date columns
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    df["Return_Date"] = pd.to_datetime(df["Return_Date"], errors="coerce")

    # 3. Fill missing Return_Reason and Return_Date for non-returned orders
    df["Return_Reason"] = df["Return_Reason"].fillna("No Return")
    # Days_to_Return is NaN for non-returned items -- fill with 0 for those
    df["Days_to_Return"] = df["Days_to_Return"].fillna(0)

    # 4. Standardize text categoricals (strip whitespace, title-case)
    str_cols = ["Product_Category", "Payment_Method", "Shipping_Method",
                "Return_Status", "Return_Reason", "User_Gender", "User_Location"]
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # 5. Encode binary target
    df["Is_Returned"] = (df["Return_Status"] == "Returned").astype(int)

    # 6. Derived date fields
    df["Order_Year"]  = df["Order_Date"].dt.year
    df["Order_Month"] = df["Order_Date"].dt.month
    df["Order_DOW"]   = df["Order_Date"].dt.dayofweek   # 0=Monday

    # 7. Clip nonsensical values
    df["User_Age"] = df["User_Age"].clip(lower=0, upper=120)
    df["Product_Price"] = df["Product_Price"].clip(lower=0)
    df["Order_Quantity"] = df["Order_Quantity"].clip(lower=1)
    df["Discount_Applied"] = df["Discount_Applied"].clip(lower=0, upper=100)

    print(f"[CLEAN] Cleaning complete. Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


# ---------------------------------------------
# SALES CALCULATION
# ---------------------------------------------

def calculate_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Add revenue, discount, and net-sales columns."""
    df = df.copy()

    # Gross revenue per order line
    df["Gross_Revenue"] = (df["Product_Price"] * df["Order_Quantity"]).round(2)

    # Discount amount in currency
    df["Discount_Amount"] = (df["Gross_Revenue"] * df["Discount_Applied"] / 100).round(2)

    # Net revenue after discount
    df["Net_Revenue"] = (df["Gross_Revenue"] - df["Discount_Amount"]).round(2)

    # Revenue lost to returns
    df["Revenue_At_Risk"] = (df["Net_Revenue"] * df["Is_Returned"]).round(2)

    print("[SALES] Revenue columns added: Gross_Revenue, Discount_Amount, Net_Revenue, Revenue_At_Risk")
    return df


# ---------------------------------------------
# DESCRIPTIVE ANALYTICS
# ---------------------------------------------

def descriptive_analytics(df: pd.DataFrame) -> dict:
    """
    Compute key summary statistics and grouped metrics.
    Returns a dict of analytics results.
    """
    results = {}

    # -- Overall KPIs --
    total_orders = len(df)
    total_returned = df["Is_Returned"].sum()
    return_rate = total_returned / total_orders * 100
    total_gross = df["Gross_Revenue"].sum()
    total_net = df["Net_Revenue"].sum()
    total_at_risk = df["Revenue_At_Risk"].sum()

    results["kpis"] = {
        "Total Orders": total_orders,
        "Total Returned": int(total_returned),
        "Return Rate (%)": round(return_rate, 2),
        "Total Gross Revenue ($)": round(total_gross, 2),
        "Total Net Revenue ($)": round(total_net, 2),
        "Revenue At Risk ($)": round(total_at_risk, 2),
        "Avg Order Value ($)": round(df["Net_Revenue"].mean(), 2),
        "Avg Discount (%)": round(df["Discount_Applied"].mean(), 2),
    }

    # -- Return Rate by Category --
    results["by_category"] = (
        df.groupby("Product_Category")
        .agg(
            Orders=("Order_ID", "count"),
            Returns=("Is_Returned", "sum"),
            Gross_Revenue=("Gross_Revenue", "sum"),
            Net_Revenue=("Net_Revenue", "sum"),
            Avg_Price=("Product_Price", "mean"),
        )
        .assign(Return_Rate=lambda x: (x["Returns"] / x["Orders"] * 100).round(2))
        .round(2)
        .sort_values("Return_Rate", ascending=False)
    )

    # -- Return Rate by Payment Method --
    results["by_payment"] = (
        df.groupby("Payment_Method")
        .agg(Orders=("Order_ID", "count"), Returns=("Is_Returned", "sum"))
        .assign(Return_Rate=lambda x: (x["Returns"] / x["Orders"] * 100).round(2))
        .sort_values("Return_Rate", ascending=False)
    )

    # -- Return Rate by Shipping Method --
    results["by_shipping"] = (
        df.groupby("Shipping_Method")
        .agg(Orders=("Order_ID", "count"), Returns=("Is_Returned", "sum"))
        .assign(Return_Rate=lambda x: (x["Returns"] / x["Orders"] * 100).round(2))
        .sort_values("Return_Rate", ascending=False)
    )

    # -- Return Rate by Gender --
    results["by_gender"] = (
        df.groupby("User_Gender")
        .agg(Orders=("Order_ID", "count"), Returns=("Is_Returned", "sum"))
        .assign(Return_Rate=lambda x: (x["Returns"] / x["Orders"] * 100).round(2))
    )

    # -- Monthly Order & Return Trend --
    results["monthly_trend"] = (
        df.groupby(["Order_Year", "Order_Month"])
        .agg(
            Orders=("Order_ID", "count"),
            Returns=("Is_Returned", "sum"),
            Net_Revenue=("Net_Revenue", "sum"),
        )
        .assign(Return_Rate=lambda x: (x["Returns"] / x["Orders"] * 100).round(2))
        .round(2)
    )

    # -- Return Reason Distribution --
    results["return_reasons"] = (
        df[df["Is_Returned"] == 1]["Return_Reason"]
        .value_counts()
        .reset_index()
        .rename(columns={"count": "Count"})
    )

    # -- Age-bin analysis --
    df["Age_Group"] = pd.cut(
        df["User_Age"],
        bins=[0, 25, 35, 45, 55, 120],
        labels=["18-25", "26-35", "36-45", "46-55", "55+"],
    )
    results["by_age_group"] = (
        df.groupby("Age_Group", observed=True)
        .agg(Orders=("Order_ID", "count"), Returns=("Is_Returned", "sum"))
        .assign(Return_Rate=lambda x: (x["Returns"] / x["Orders"] * 100).round(2))
    )

    # -- Top 10 high-risk locations --
    results["top_locations"] = (
        df.groupby("User_Location")
        .agg(Orders=("Order_ID", "count"), Returns=("Is_Returned", "sum"))
        .assign(Return_Rate=lambda x: (x["Returns"] / x["Orders"] * 100).round(2))
        .sort_values("Return_Rate", ascending=False)
        .head(10)
    )

    return results


def print_analytics_report(results: dict) -> str:
    """Pretty-print and save descriptive analytics."""
    lines = []
    lines.append("=" * 70)
    lines.append("  DESCRIPTIVE ANALYTICS REPORT -- E-Commerce Returns")
    lines.append("=" * 70)

    lines.append("\n-- Overall KPIs --")
    for k, v in results["kpis"].items():
        if isinstance(v, float):
            lines.append(f"  {k:<35} {v:>12,.2f}")
        else:
            lines.append(f"  {k:<35} {v:>12,}")

    lines.append("\n-- Return Rate by Product Category --")
    lines.append(results["by_category"].to_string())

    lines.append("\n-- Return Rate by Payment Method --")
    lines.append(results["by_payment"].to_string())

    lines.append("\n-- Return Rate by Shipping Method --")
    lines.append(results["by_shipping"].to_string())

    lines.append("\n-- Return Rate by Gender --")
    lines.append(results["by_gender"].to_string())

    lines.append("\n-- Return Reasons (Returned Orders Only) --")
    lines.append(results["return_reasons"].to_string(index=False))

    lines.append("\n-- Return Rate by Age Group --")
    lines.append(results["by_age_group"].to_string())

    lines.append("\n-- Top 10 High-Risk Locations --")
    lines.append(results["top_locations"].to_string())

    lines.append("\n-- Monthly Trend (last 12 months shown) --")
    lines.append(results["monthly_trend"].tail(12).to_string())

    lines.append("\n" + "=" * 70)
    text = "\n".join(lines)
    print(text)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n[SAVED] Analytics report -> {REPORT_PATH}")
    return text


if __name__ == "__main__":
    from src.s01_data_loading_and_quality import load_data
    df_raw = load_data()
    df_clean = clean_data(df_raw)
    df_sales = calculate_sales(df_clean)
    results = descriptive_analytics(df_sales)
    print_analytics_report(results)
