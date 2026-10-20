"""
Module 7: Return-Risk Prediction Application (CLI)
AI-Powered E-Commerce Return & Refund Risk Analytics

Usage:
    python src/s07_prediction_app.py
    python src/s07_prediction_app.py --batch path/to/new_orders.csv
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

MODEL_DIR = "outputs/models"


# ---------------------------------------------
# LOAD ARTIFACTS
# ---------------------------------------------

def load_artifacts():
    """Load trained model and feature names."""
    # Find available model file
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    if not model_files:
        raise FileNotFoundError(
            f"No model found in {MODEL_DIR}. Run main.py first."
        )
    # Prefer tuned models
    preferred = [f for f in model_files if "tuned" in f or "random_forest" in f or "gradient" in f]
    model_file = preferred[0] if preferred else model_files[0]
    model_path = os.path.join(MODEL_DIR, model_file)
    model = joblib.load(model_path)

    feat_path = os.path.join(MODEL_DIR, "feature_names.txt")
    with open(feat_path, "r") as f:
        feature_names = [line.strip() for line in f if line.strip()]

    print(f"[APP] Model loaded    : {model_file}")
    print(f"[APP] Features loaded : {len(feature_names)} features")
    return model, feature_names


# ---------------------------------------------
# FEATURE PREPARATION (mirrors feature engineering)
# ---------------------------------------------

CATEGORY_MAP    = {"Clothing": 0, "Books": 1, "Toys": 2, "Electronics": 3, "Home": 4}
PAYMENT_MAP     = {"Credit Card": 0, "Debit Card": 1, "Gift Card": 2, "PayPal": 3}
SHIPPING_MAP    = {"Express": 0, "Next-Day": 1, "Standard": 2}
GENDER_MAP      = {"Female": 0, "Male": 1}
PRICE_TIER_MAP  = {"Budget": 0, "Luxury": 1, "Mid": 2, "Premium": 3}
DISC_TIER_MAP   = {"High": 0, "Low": 1, "Medium": 2, "Very High": 3}
AGE_GROUP_MAP   = {"18-25": 0, "26-35": 1, "36-45": 2, "46-55": 3, "55+": 4}
SEASON_MAP      = {"Autumn": 0, "Spring": 1, "Summer": 2, "Winter": 3}


def build_feature_row(order: dict) -> dict:
    """
    Convert a raw order dict into a model-ready feature dict.
    Expects keys matching the interactive prompt inputs.
    """
    price     = float(order["Product_Price"])
    qty       = int(order["Order_Quantity"])
    age       = int(order["User_Age"])
    discount  = float(order["Discount_Applied"])
    month     = int(order.get("Order_Month", 6))
    dow       = int(order.get("Order_DOW", 2))
    category  = str(order["Product_Category"])
    payment   = str(order["Payment_Method"])
    shipping  = str(order["Shipping_Method"])
    gender    = str(order["User_Gender"])

    gross_rev = price * qty
    discount_amt = gross_rev * discount / 100
    net_rev = gross_rev - discount_amt
    qty_x_price = qty * price
    net_per_unit = net_rev / qty

    # Price tier
    if price <= 100:
        price_tier = "Budget"
    elif price <= 250:
        price_tier = "Mid"
    elif price <= 400:
        price_tier = "Premium"
    else:
        price_tier = "Luxury"

    # Discount tier
    if discount <= 10:
        disc_tier = "Low"
    elif discount <= 25:
        disc_tier = "Medium"
    elif discount <= 50:
        disc_tier = "High"
    else:
        disc_tier = "Very High"

    # Age group
    if age <= 25:
        age_group = "18-25"
    elif age <= 35:
        age_group = "26-35"
    elif age <= 45:
        age_group = "36-45"
    elif age <= 55:
        age_group = "46-55"
    else:
        age_group = "55+"

    # Season
    if month in [12, 1, 2]:
        season = "Winter"
    elif month in [3, 4, 5]:
        season = "Spring"
    elif month in [6, 7, 8]:
        season = "Summer"
    else:
        season = "Autumn"

    is_high_value    = 1 if net_rev > 900 else 0     # approximate 75th pct
    is_weekend       = 1 if dow >= 5 else 0
    is_fast_shipping = 1 if shipping in ["Express", "Next-Day"] else 0
    is_high_discount = 1 if discount > 30 else 0
    location_freq    = 0.01   # default for new/unknown location

    features = {
        "Product_Price":       price,
        "Order_Quantity":      qty,
        "User_Age":            age,
        "Discount_Applied":    discount,
        "Gross_Revenue":       round(gross_rev, 2),
        "Net_Revenue":         round(net_rev, 2),
        "Qty_x_Price":         round(qty_x_price, 2),
        "Net_Per_Unit":        round(net_per_unit, 2),
        "Location_Freq":       location_freq,
        "Order_Month":         month,
        "Order_DOW":           dow,
        "Is_High_Value":       is_high_value,
        "Is_Weekend":          is_weekend,
        "Is_Fast_Shipping":    is_fast_shipping,
        "Is_High_Discount":    is_high_discount,
        "Product_Category_enc":  CATEGORY_MAP.get(category, 0),
        "Payment_Method_enc":    PAYMENT_MAP.get(payment, 0),
        "Shipping_Method_enc":   SHIPPING_MAP.get(shipping, 0),
        "User_Gender_enc":       GENDER_MAP.get(gender, 0),
        "Price_Tier_enc":        PRICE_TIER_MAP.get(price_tier, 0),
        "Discount_Tier_enc":     DISC_TIER_MAP.get(disc_tier, 0),
        "Age_Group_enc":         AGE_GROUP_MAP.get(age_group, 0),
        "Season_enc":            SEASON_MAP.get(season, 0),
    }
    return features


# ---------------------------------------------
# RISK SCORING
# ---------------------------------------------

def risk_label(prob: float) -> tuple:
    """Convert probability to risk label and recommended action."""
    if prob >= 0.75:
        return "? VERY HIGH RISK", "Immediate intervention: offer exchange, add review checklist"
    elif prob >= 0.60:
        return "? HIGH RISK",      "Proactive outreach: send product guide/video before delivery"
    elif prob >= 0.45:
        return "? MODERATE RISK",  "Monitor: include detailed instructions and follow-up email"
    elif prob >= 0.30:
        return "? LOW RISK",       "Standard processing -- no special action required"
    else:
        return "[OK] VERY LOW RISK",  "Green-light for fast fulfillment"


def predict_single(model, feature_names: list, order: dict) -> dict:
    """Run prediction for one order and return result dict."""
    feat_row = build_feature_row(order)
    X = pd.DataFrame([feat_row])[feature_names]
    prob  = model.predict_proba(X)[0, 1]
    label = model.predict(X)[0]
    risk_lbl, action = risk_label(prob)
    return {
        "return_probability": round(prob * 100, 2),
        "predicted_class": "Returned" if label == 1 else "Not Returned",
        "risk_level": risk_lbl,
        "recommended_action": action,
    }


def predict_batch(model, feature_names: list, df_new: pd.DataFrame) -> pd.DataFrame:
    """Score a batch of new orders from a DataFrame."""
    rows = []
    for _, row in df_new.iterrows():
        feat = build_feature_row(row.to_dict())
        rows.append(feat)

    X = pd.DataFrame(rows)[feature_names]
    probs = model.predict_proba(X)[:, 1]
    labels = model.predict(X)

    df_out = df_new.copy().reset_index(drop=True)
    df_out["Return_Probability_%"] = (probs * 100).round(2)
    df_out["Predicted_Return"]     = ["Returned" if l == 1 else "Not Returned" for l in labels]
    df_out["Risk_Level"]           = [risk_label(p)[0] for p in probs]
    df_out["Recommended_Action"]   = [risk_label(p)[1] for p in probs]
    return df_out


# ---------------------------------------------
# CLI INTERACTIVE APP
# ---------------------------------------------

BANNER = """
+==============================================================+
|   AI-Powered E-Commerce Return & Refund Risk Predictor       |
|   Predict return probability for any new order               |
+==============================================================+
"""

MENU = """
Options:
  [1] Predict return risk for a single order (manual input)
  [2] Predict return risk for a batch (CSV file)
  [3] Show sample prediction with example data
  [Q] Quit
"""


def _prompt(label: str, options: list = None, cast=str, default=None):
    """Interactive prompt with validation."""
    while True:
        hint = f" [{'/'.join(options)}]" if options else ""
        val = input(f"  {label}{hint}: ").strip()
        if not val and default is not None:
            return default
        if options and val not in options:
            print(f"    ? Please enter one of: {options}")
            continue
        try:
            return cast(val)
        except ValueError:
            print(f"    ? Expected a number. Try again.")


def interactive_single(model, feature_names: list):
    """Prompt user for order details and show prediction."""
    print("\n-- Enter Order Details --")
    order = {
        "Product_Category": _prompt("Product Category",
                                    ["Clothing", "Books", "Toys", "Electronics", "Home"]),
        "Product_Price":    _prompt("Product Price ($)", cast=float),
        "Order_Quantity":   _prompt("Order Quantity", cast=int),
        "Discount_Applied": _prompt("Discount Applied (%)", cast=float),
        "Payment_Method":   _prompt("Payment Method",
                                    ["Credit Card", "Debit Card", "Gift Card", "PayPal"]),
        "Shipping_Method":  _prompt("Shipping Method",
                                    ["Express", "Next-Day", "Standard"]),
        "User_Age":         _prompt("Customer Age", cast=int),
        "User_Gender":      _prompt("Customer Gender", ["Male", "Female"]),
        "Order_Month":      _prompt("Order Month (1-12)", cast=int, default=6),
        "Order_DOW":        _prompt("Day of Week (0=Mon...6=Sun)", cast=int, default=2),
    }
    result = predict_single(model, feature_names, order)

    print("\n" + "-" * 55)
    print(f"  RETURN PROBABILITY  : {result['return_probability']}%")
    print(f"  PREDICTED STATUS    : {result['predicted_class']}")
    print(f"  RISK LEVEL          : {result['risk_level']}")
    print(f"  RECOMMENDATION      : {result['recommended_action']}")
    print("-" * 55)


def run_sample_prediction(model, feature_names: list):
    """Demonstrate with a hard-coded example order."""
    sample_order = {
        "Product_Category": "Clothing",
        "Product_Price":    389.99,
        "Order_Quantity":   3,
        "Discount_Applied": 42.5,
        "Payment_Method":   "Credit Card",
        "Shipping_Method":  "Next-Day",
        "User_Age":         28,
        "User_Gender":      "Female",
        "Order_Month":      12,
        "Order_DOW":        6,
    }
    print("\n-- Sample Order --")
    for k, v in sample_order.items():
        print(f"  {k:<25}: {v}")

    result = predict_single(model, feature_names, sample_order)

    print("\n-- Prediction Result --")
    print(f"  RETURN PROBABILITY  : {result['return_probability']}%")
    print(f"  PREDICTED STATUS    : {result['predicted_class']}")
    print(f"  RISK LEVEL          : {result['risk_level']}")
    print(f"  RECOMMENDATION      : {result['recommended_action']}")
    print("-" * 55)


def run_batch_prediction(model, feature_names: list, csv_path: str):
    """Score a CSV file of orders and save results."""
    print(f"\n[APP] Loading batch file: {csv_path}")
    df_new = pd.read_csv(csv_path)
    df_out = predict_batch(model, feature_names, df_new)

    out_path = csv_path.replace(".csv", "_predictions.csv")
    df_out.to_csv(out_path, index=False)
    print(f"[APP] Predictions saved -> {out_path}")
    print(df_out[["Return_Probability_%", "Predicted_Return", "Risk_Level"]].head(10).to_string())
    return df_out


def run_cli():
    """Main CLI loop."""
    model, feature_names = load_artifacts()
    print(BANNER)

    while True:
        print(MENU)
        choice = input("Your choice: ").strip().upper()

        if choice == "1":
            interactive_single(model, feature_names)
        elif choice == "2":
            path = input("  Enter CSV file path: ").strip()
            if os.path.exists(path):
                run_batch_prediction(model, feature_names, path)
            else:
                print(f"  ? File not found: {path}")
        elif choice == "3":
            run_sample_prediction(model, feature_names)
        elif choice == "Q":
            print("\n[APP] Goodbye!\n")
            break
        else:
            print("  ? Invalid choice. Enter 1, 2, 3 or Q.")


# ---------------------------------------------
# ENTRY POINT
# ---------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, ".")
    parser = argparse.ArgumentParser(
        description="Return Risk Prediction App"
    )
    parser.add_argument(
        "--batch", type=str, default=None,
        help="Path to a CSV file for batch prediction"
    )
    args = parser.parse_args()

    if args.batch:
        mdl, feat_names = load_artifacts()
        run_batch_prediction(mdl, feat_names, args.batch)
    else:
        run_cli()
