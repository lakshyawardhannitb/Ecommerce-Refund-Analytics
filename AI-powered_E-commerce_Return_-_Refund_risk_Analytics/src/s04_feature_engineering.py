"""
Module 4: Feature Engineering
AI-Powered E-Commerce Return & Refund Risk Analytics
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all features required for ML model training.
    Returns a copy with new feature columns added.
    """
    df = df.copy()

    # -- 1. Price tier --
    df["Price_Tier"] = pd.cut(
        df["Product_Price"],
        bins=[0, 100, 250, 400, np.inf],
        labels=["Budget", "Mid", "Premium", "Luxury"],
    ).astype(str)

    # -- 2. Discount tier --
    df["Discount_Tier"] = pd.cut(
        df["Discount_Applied"],
        bins=[-1, 10, 25, 50, 100],
        labels=["Low", "Medium", "High", "Very High"],
    ).astype(str)

    # -- 3. Age group --
    df["Age_Group"] = pd.cut(
        df["User_Age"],
        bins=[0, 25, 35, 45, 55, 120],
        labels=["18-25", "26-35", "36-45", "46-55", "55+"],
    ).astype(str)

    # -- 4. Quantity x Price interaction --
    df["Qty_x_Price"] = df["Order_Quantity"] * df["Product_Price"]

    # -- 5. Net revenue per unit --
    df["Net_Per_Unit"] = (df["Net_Revenue"] / df["Order_Quantity"]).round(2)

    # -- 6. Is high-value order --
    df["Is_High_Value"] = (df["Net_Revenue"] > df["Net_Revenue"].quantile(0.75)).astype(int)

    # -- 7. Is weekend order --
    df["Is_Weekend"] = (df["Order_DOW"] >= 5).astype(int)

    # -- 8. Season from order month --
    def month_to_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"

    df["Season"] = df["Order_Month"].apply(month_to_season)

    # -- 9. Express/Next-Day flag --
    df["Is_Fast_Shipping"] = (
        df["Shipping_Method"].isin(["Express", "Next-Day"])
    ).astype(int)

    # -- 10. High-discount flag --
    df["Is_High_Discount"] = (df["Discount_Applied"] > 30).astype(int)

    # -- 11. User location encoded (frequency encoding) --
    loc_freq = df["User_Location"].value_counts(normalize=True)
    df["Location_Freq"] = df["User_Location"].map(loc_freq).round(5)

    print(f"[FE] Feature engineering complete. New shape: {df.shape[0]:,} x {df.shape[1]}")
    return df


def prepare_ml_features(df: pd.DataFrame):
    """
    Encode and select features for ML training.
    Returns: X (features DataFrame), y (target Series), feature_names (list)
    """
    df = df.copy()

    # -- Label-encode categoricals --
    cat_features = [
        "Product_Category", "Payment_Method", "Shipping_Method",
        "User_Gender", "Price_Tier", "Discount_Tier", "Age_Group",
        "Season",
    ]

    le = LabelEncoder()
    for col in cat_features:
        if col in df.columns:
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    # -- Select final feature set --
    feature_cols = [
        # Numeric
        "Product_Price",
        "Order_Quantity",
        "User_Age",
        "Discount_Applied",
        "Gross_Revenue",
        "Net_Revenue",
        "Qty_x_Price",
        "Net_Per_Unit",
        "Location_Freq",
        "Order_Month",
        "Order_DOW",
        # Binary flags
        "Is_High_Value",
        "Is_Weekend",
        "Is_Fast_Shipping",
        "Is_High_Discount",
        # Encoded categoricals
        "Product_Category_enc",
        "Payment_Method_enc",
        "Shipping_Method_enc",
        "User_Gender_enc",
        "Price_Tier_enc",
        "Discount_Tier_enc",
        "Age_Group_enc",
        "Season_enc",
    ]

    # Keep only available columns
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df["Is_Returned"].copy()

    print(f"[FE] ML feature matrix: {X.shape[0]:,} samples x {X.shape[1]} features")
    print(f"[FE] Target balance -- Returned: {y.sum():,} | Not Returned: {(y == 0).sum():,}")

    return X, y, feature_cols


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.s01_data_loading_and_quality import load_data
    from src.s02_data_cleaning_and_analytics import clean_data, calculate_sales

    df = calculate_sales(clean_data(load_data()))
    df_fe = engineer_features(df)
    X, y, feat_names = prepare_ml_features(df_fe)
    print(X.head(3))
    print(f"\nFeatures ({len(feat_names)}):")
    for f in feat_names:
        print(f"  - {f}")
