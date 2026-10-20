"""
Module 6: Explainable AI (SHAP) + Business Insights
AI-Powered E-Commerce Return & Refund Risk Analytics
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import os
import warnings
warnings.filterwarnings("ignore")

PLOT_DIR   = "outputs/plots"
REPORT_DIR = "outputs/reports"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ---------------------------------------------
# SHAP EXPLAINABILITY
# ---------------------------------------------

def run_shap_analysis(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                      feature_names: list, model_name: str = "Best Model"):
    """
    Compute SHAP values and produce:
      - Summary (beeswarm) plot
      - Bar summary plot
      - Waterfall plot for a single high-risk prediction
      - Dependence plots for top 3 features
    """
    print(f"\n[SHAP] Computing SHAP values for {model_name}...")

    # Use TreeExplainer for tree-based models
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        # Normalise to 2-D (samples x features) for class 1 = "Returned"
        sv_raw = np.array(shap_values)
        if sv_raw.ndim == 3:
            # shape (samples, features, classes) — take class-1 slice
            sv = sv_raw[:, :, 1]
        elif sv_raw.ndim == 2:
            sv = sv_raw
        else:
            # list of arrays  [class0, class1]
            sv = sv_raw[1] if len(sv_raw) == 2 else sv_raw[0]
    except Exception as e:
        print(f"  [SHAP] TreeExplainer failed ({e}), falling back to KernelExplainer...")
        background = shap.sample(X_train, 100, random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        raw = explainer.shap_values(X_test.iloc[:200])
        sv_raw = np.array(raw)
        sv = sv_raw[:, :, 1] if sv_raw.ndim == 3 else (sv_raw[1] if isinstance(raw, list) else sv_raw)

    feat_arr = list(feature_names)

    # -- 1. Beeswarm summary plot --
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv, X_test, feature_names=feat_arr,
                      show=False, plot_size=None)
    plt.title(f"SHAP Summary Plot -- {model_name}", fontweight="bold", fontsize=12)
    path1 = os.path.join(PLOT_DIR, "18_shap_summary_beeswarm.png")
    plt.savefig(path1, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  [PLOT] Saved -> {path1}")

    # -- 2. Bar summary (mean |SHAP|) --
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    shap.summary_plot(sv, X_test, feature_names=feat_arr,
                      plot_type="bar", show=False, plot_size=None)
    plt.title(f"SHAP Feature Importance (|Mean SHAP|) -- {model_name}",
              fontweight="bold", fontsize=12)
    path2 = os.path.join(PLOT_DIR, "19_shap_bar_importance.png")
    plt.savefig(path2, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  [PLOT] Saved -> {path2}")

    # -- 3. Waterfall for highest-risk prediction --
    try:
        sv_arr = np.array(sv)
        risk_scores = sv_arr.sum(axis=1)
        high_risk_idx = int(np.argmax(risk_scores))
        # Resolve base value for class 1
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            base_val = float(np.array(ev).flat[1]) if len(ev) > 1 else float(np.array(ev).flat[0])
        else:
            base_val = float(ev)
        exp = shap.Explanation(
            values=sv_arr[high_risk_idx],
            base_values=base_val,
            data=X_test.iloc[high_risk_idx].values if isinstance(X_test, pd.DataFrame)
                 else X_test[high_risk_idx],
            feature_names=feat_arr,
        )
        shap.plots.waterfall(exp, show=False)
        plt.title(f"SHAP Waterfall -- Highest-Risk Order (idx={high_risk_idx})",
                  fontweight="bold", fontsize=11)
        path3 = os.path.join(PLOT_DIR, "20_shap_waterfall_high_risk.png")
        plt.savefig(path3, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"  [PLOT] Saved -> {path3}")
    except Exception as e:
        print(f"  [SHAP] Waterfall skipped: {e}")

    # -- 4. SHAP dependence plots for top 3 features --
    sv_arr = np.array(sv)
    mean_abs = np.abs(sv_arr).mean(axis=0)
    top3_idx = np.argsort(mean_abs)[::-1][:3]
    for rank, idx in enumerate(top3_idx, 1):
        feat = feat_arr[int(idx)]
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        X_test_df = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test, columns=feat_arr)
        shap.dependence_plot(
            ind=int(idx),
            shap_values=sv_arr,
            features=X_test_df,
            feature_names=feat_arr,
            ax=ax4,
            show=False,
        )
        ax4.set_title(f"SHAP Dependence -- {feat}", fontweight="bold")
        path4 = os.path.join(PLOT_DIR, f"21_shap_dependence_top{rank}_{feat.lower()}.png")
        fig4.savefig(path4, bbox_inches="tight", dpi=150)
        plt.close(fig4)
        print(f"  [PLOT] Saved -> {path4}")

    print("[SHAP] Analysis complete.")
    return sv, explainer


# ---------------------------------------------
# BUSINESS INSIGHTS & RECOMMENDATIONS
# ---------------------------------------------

def generate_business_insights(df: pd.DataFrame, shap_values, feature_names: list) -> str:
    """
    Generate AI-assisted business recommendations based on:
      - Descriptive stats
      - SHAP feature importances
      - Return rate patterns
    Returns a formatted text report.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("  AI-ASSISTED BUSINESS INSIGHTS & RECOMMENDATIONS")
    lines.append("  E-Commerce Return & Refund Risk Analytics")
    lines.append("=" * 70)

    total = len(df)
    n_returned = df["Is_Returned"].sum()
    return_rate = n_returned / total * 100
    revenue_at_risk = df["Revenue_At_Risk"].sum()
    total_net_rev = df["Net_Revenue"].sum()
    rev_loss_pct = revenue_at_risk / total_net_rev * 100

    lines.append(f"""
+======================================================+
|  EXECUTIVE SUMMARY                                   |
+======================================================+
|  Total Orders        : {total:>10,}                    |
|  Total Returned      : {int(n_returned):>10,}                    |
|  Overall Return Rate : {return_rate:>9.1f}%                    |
|  Revenue at Risk     : ${revenue_at_risk:>12,.2f}              |
|  % of Net Revenue    : {rev_loss_pct:>9.1f}%                    |
+======================================================+
""")

    # -- SHAP-driven feature ranking --
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_series = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)

    lines.append("-- Top Return-Risk Drivers (SHAP) --")
    for i, (feat, imp) in enumerate(importance_series.head(8).items(), 1):
        lines.append(f"  {i}. {feat:<35} SHAP Importance = {imp:.4f}")

    # -- Category analysis --
    cat_rates = (
        df.groupby("Product_Category")["Is_Returned"]
        .mean().mul(100).sort_values(ascending=False)
    )
    lines.append("\n-- Return Rate by Category --")
    for cat, rate in cat_rates.items():
        flag = "! HIGH RISK" if rate > 55 else ("OK" if rate < 45 else "~ MODERATE")
        lines.append(f"  {cat:<20} {rate:.1f}%  {flag}")

    # -- Payment method analysis --
    pay_rates = (
        df.groupby("Payment_Method")["Is_Returned"]
        .mean().mul(100).sort_values(ascending=False)
    )
    lines.append("\n-- Return Rate by Payment Method --")
    for pm, rate in pay_rates.items():
        lines.append(f"  {pm:<20} {rate:.1f}%")

    # -- High-risk customer segment --
    high_risk_segment = df[df["Is_Returned"] == 1]
    avg_age_returned = high_risk_segment["User_Age"].mean()
    avg_discount_returned = high_risk_segment["Discount_Applied"].mean()
    top_reason = high_risk_segment["Return_Reason"].value_counts().index[0]

    lines.append(f"""
-- High-Risk Customer Profile --
  Average Age (returned orders) : {avg_age_returned:.1f} years
  Average Discount (returned)   : {avg_discount_returned:.1f}%
  Most Common Return Reason     : {top_reason}
""")

    # -- Actionable Recommendations --
    lines.append("=" * 70)
    lines.append("  ACTIONABLE RECOMMENDATIONS")
    lines.append("=" * 70)

    recommendations = [
        (
            "1. PRODUCT QUALITY ALERTS",
            f"Categories with return rates above 55% should trigger automatic quality "
            f"reviews. Focus on: {', '.join(cat_rates[cat_rates > 55].index.tolist() or ['None currently'])}.",
        ),
        (
            "2. SMART DISCOUNT POLICY",
            f"Data shows high discounts correlate with higher returns. Consider capping "
            f"discounts at 25% for high-return-risk categories. Currently, returned orders "
            f"average {avg_discount_returned:.1f}% discount vs. "
            f"{df[df['Is_Returned']==0]['Discount_Applied'].mean():.1f}% for kept orders.",
        ),
        (
            "3. TARGETED PRE-PURCHASE INTERVENTIONS",
            "For orders flagged as high-risk by the ML model (probability > 0.6), "
            "trigger enhanced product descriptions, size guides, or a confirmation step "
            "to reduce impulsive purchases.",
        ),
        (
            "4. RETURN FRICTION STRATEGY",
            f"The top return reason is '{top_reason}'. Address this through better "
            "product imagery, accurate sizing/specifications, and stricter QA before dispatch.",
        ),
        (
            "5. PERSONALIZED RETURN POLICY",
            "Implement tiered return policies: customers with historically high return "
            "rates (>3 returns) should face a restocking fee or a shorter return window "
            "to reduce abuse.",
        ),
        (
            "6. PAYMENT METHOD RISK SCORING",
            f"Highest-risk payment method: {pay_rates.index[0]} ({pay_rates.iloc[0]:.1f}% return rate). "
            "Consider additional verification steps or targeted communication for these orders.",
        ),
        (
            "7. REVENUE RECOVERY PROGRAM",
            f"${revenue_at_risk:,.0f} in net revenue is at risk from returns. Implement "
            "proactive outreach (email/SMS) for high-probability-return orders to offer "
            "alternatives (exchange, credit) before a full return is requested.",
        ),
        (
            "8. REAL-TIME RISK SCORING AT CHECKOUT",
            "Deploy the trained ML model as a real-time scoring API. Flag orders with "
            "return probability > 0.65 for human review or automated intervention.",
        ),
    ]

    for title, body in recommendations:
        lines.append(f"\n  {title}")
        lines.append(f"  {'-' * 60}")
        # Word-wrap body at ~65 chars
        import textwrap
        for wrapped_line in textwrap.wrap(body, width=65):
            lines.append(f"  {wrapped_line}")

    lines.append("\n" + "=" * 70)
    text = "\n".join(lines)
    print(text)

    path = os.path.join(REPORT_DIR, "business_insights_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n[SAVED] Business insights report -> {path}")
    return text


if __name__ == "__main__":
    import sys, joblib
    sys.path.insert(0, ".")
    from src.s01_data_loading_and_quality import load_data
    from src.s02_data_cleaning_and_analytics import clean_data, calculate_sales
    from src.s04_feature_engineering import engineer_features, prepare_ml_features
    from src.s05_ml_pipeline import split_data

    df = calculate_sales(clean_data(load_data()))
    df_fe = engineer_features(df)
    X, y, feat_names = prepare_ml_features(df_fe)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = joblib.load("outputs/models/random_forest_(tuned).pkl")
    sv, explainer = run_shap_analysis(model, X_train, X_test, feat_names)
    generate_business_insights(df_fe, sv, feat_names)
