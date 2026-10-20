"""
Module 3: Exploratory Data Analysis (EDA) & Data Visualization
AI-Powered E-Commerce Return & Refund Risk Analytics
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os

PLOT_DIR = "outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# -- Shared style --
PALETTE = "Set2"
PRIMARY  = "#3b82d4"
DANGER   = "#e74c3c"
SUCCESS  = "#2ecc71"
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.05)

def _save(fig: plt.Figure, name: str):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [PLOT] Saved -> {path}")


# ---------------------------------------------
# 1. TARGET DISTRIBUTION
# ---------------------------------------------
def plot_target_distribution(df: pd.DataFrame):
    counts = df["Return_Status"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Return Status Distribution", fontsize=14, fontweight="bold")

    # Bar
    axes[0].bar(counts.index, counts.values, color=[SUCCESS, DANGER], edgecolor="white")
    for i, (label, val) in enumerate(counts.items()):
        axes[0].text(i, val + 30, f"{val:,}", ha="center", fontsize=11)
    axes[0].set_ylabel("Number of Orders")
    axes[0].set_title("Count")

    # Pie
    axes[1].pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=[SUCCESS, DANGER],
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    axes[1].set_title("Proportion")
    _save(fig, "01_target_distribution.png")


# ---------------------------------------------
# 2. RETURN RATE BY CATEGORY
# ---------------------------------------------
def plot_return_by_category(df: pd.DataFrame):
    grp = (
        df.groupby("Product_Category")["Is_Returned"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"Is_Returned": "Return_Rate"})
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(grp["Product_Category"], grp["Return_Rate"],
                   color=sns.color_palette(PALETTE, len(grp)))
    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.set_xlabel("Return Rate (%)")
    ax.set_title("Return Rate by Product Category", fontweight="bold")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.invert_yaxis()
    _save(fig, "02_return_rate_by_category.png")


# ---------------------------------------------
# 3. RETURN RATE BY PAYMENT METHOD
# ---------------------------------------------
def plot_return_by_payment(df: pd.DataFrame):
    grp = (
        df.groupby("Payment_Method")["Is_Returned"]
        .mean().mul(100).sort_values(ascending=False)
        .reset_index().rename(columns={"Is_Returned": "Return_Rate"})
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=grp, x="Payment_Method", y="Return_Rate",
                palette=PALETTE, ax=ax)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha="center", fontsize=10)
    ax.set_title("Return Rate by Payment Method", fontweight="bold")
    ax.set_ylabel("Return Rate (%)")
    _save(fig, "03_return_rate_by_payment.png")


# ---------------------------------------------
# 4. RETURN RATE BY SHIPPING METHOD
# ---------------------------------------------
def plot_return_by_shipping(df: pd.DataFrame):
    grp = (
        df.groupby("Shipping_Method")["Is_Returned"]
        .mean().mul(100).sort_values(ascending=False)
        .reset_index().rename(columns={"Is_Returned": "Return_Rate"})
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=grp, x="Shipping_Method", y="Return_Rate",
                palette=PALETTE, ax=ax)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha="center", fontsize=10)
    ax.set_title("Return Rate by Shipping Method", fontweight="bold")
    ax.set_ylabel("Return Rate (%)")
    _save(fig, "04_return_rate_by_shipping.png")


# ---------------------------------------------
# 5. RETURN REASONS (PIE)
# ---------------------------------------------
def plot_return_reasons(df: pd.DataFrame):
    reasons = df[df["Is_Returned"] == 1]["Return_Reason"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        reasons.values,
        labels=reasons.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette(PALETTE, len(reasons)),
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    ax.set_title("Return Reasons Distribution", fontweight="bold", fontsize=13)
    _save(fig, "05_return_reasons.png")


# ---------------------------------------------
# 6. PRICE DISTRIBUTION (RETURNED vs NOT)
# ---------------------------------------------
def plot_price_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color in [("Returned", DANGER), ("Not Returned", SUCCESS)]:
        subset = df[df["Return_Status"] == label]["Product_Price"]
        ax.hist(subset, bins=40, alpha=0.65, label=label,
                color=color, edgecolor="white")
    ax.set_xlabel("Product Price ($)")
    ax.set_ylabel("Frequency")
    ax.set_title("Product Price Distribution: Returned vs Not Returned",
                 fontweight="bold")
    ax.legend()
    _save(fig, "06_price_distribution.png")


# ---------------------------------------------
# 7. AGE DISTRIBUTION (RETURNED vs NOT)
# ---------------------------------------------
def plot_age_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color in [("Returned", DANGER), ("Not Returned", SUCCESS)]:
        subset = df[df["Return_Status"] == label]["User_Age"]
        ax.hist(subset, bins=30, alpha=0.65, label=label,
                color=color, edgecolor="white")
    ax.set_xlabel("User Age")
    ax.set_ylabel("Frequency")
    ax.set_title("User Age Distribution: Returned vs Not Returned",
                 fontweight="bold")
    ax.legend()
    _save(fig, "07_age_distribution.png")


# ---------------------------------------------
# 8. MONTHLY RETURN TREND
# ---------------------------------------------
def plot_monthly_trend(df: pd.DataFrame):
    df["YearMonth"] = df["Order_Date"].dt.to_period("M")
    monthly = (
        df.groupby("YearMonth")
        .agg(Orders=("Order_ID", "count"), Returns=("Is_Returned", "sum"))
        .assign(Return_Rate=lambda x: x["Returns"] / x["Orders"] * 100)
        .reset_index()
    )
    monthly["YearMonth_str"] = monthly["YearMonth"].astype(str)

    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax2 = ax1.twinx()

    ax1.bar(monthly["YearMonth_str"], monthly["Orders"],
            color=PRIMARY, alpha=0.5, label="Total Orders")
    ax2.plot(monthly["YearMonth_str"], monthly["Return_Rate"],
             color=DANGER, marker="o", linewidth=2, label="Return Rate %")

    ax1.set_xlabel("Month")
    ax1.set_ylabel("Number of Orders", color=PRIMARY)
    ax2.set_ylabel("Return Rate (%)", color=DANGER)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.set_title("Monthly Order Volume & Return Rate Trend", fontweight="bold")

    # Rotate x labels
    plt.xticks(rotation=45, ha="right")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    _save(fig, "08_monthly_trend.png")


# ---------------------------------------------
# 9. CORRELATION HEATMAP
# ---------------------------------------------
def plot_correlation_heatmap(df: pd.DataFrame):
    num_cols = ["Product_Price", "Order_Quantity", "Days_to_Return",
                "User_Age", "Discount_Applied", "Gross_Revenue",
                "Net_Revenue", "Is_Returned"]
    corr = df[[c for c in num_cols if c in df.columns]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontweight="bold")
    _save(fig, "09_correlation_heatmap.png")


# ---------------------------------------------
# 10. DISCOUNT vs RETURN RATE (BOX PLOT)
# ---------------------------------------------
def plot_discount_vs_return(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="Return_Status", y="Discount_Applied",
                palette=[SUCCESS, DANGER], ax=ax)
    ax.set_title("Discount Applied vs Return Status", fontweight="bold")
    ax.set_ylabel("Discount (%)")
    _save(fig, "10_discount_vs_return.png")


# ---------------------------------------------
# 11. ORDER QUANTITY vs RETURN RATE
# ---------------------------------------------
def plot_quantity_vs_return(df: pd.DataFrame):
    grp = (
        df.groupby("Order_Quantity")["Is_Returned"]
        .mean().mul(100).reset_index()
        .rename(columns={"Is_Returned": "Return_Rate"})
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(grp["Order_Quantity"], grp["Return_Rate"],
           color=PRIMARY, edgecolor="white")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlabel("Order Quantity")
    ax.set_ylabel("Return Rate (%)")
    ax.set_title("Order Quantity vs Return Rate", fontweight="bold")
    _save(fig, "11_quantity_vs_return.png")


# ---------------------------------------------
# 12. GENDER vs CATEGORY HEATMAP (RETURN RATE)
# ---------------------------------------------
def plot_gender_category_heatmap(df: pd.DataFrame):
    pivot = (
        df.groupby(["User_Gender", "Product_Category"])["Is_Returned"]
        .mean()
        .mul(100)
        .unstack()
        .round(1)
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Return Rate (%)"})
    ax.set_title("Return Rate (%) by Gender x Product Category",
                 fontweight="bold")
    _save(fig, "12_gender_category_heatmap.png")


# ---------------------------------------------
# 13. REVENUE AT RISK BY CATEGORY
# ---------------------------------------------
def plot_revenue_at_risk(df: pd.DataFrame):
    grp = (
        df.groupby("Product_Category")["Revenue_At_Risk"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(grp["Product_Category"], grp["Revenue_At_Risk"],
                  color=sns.color_palette(PALETTE, len(grp)), edgecolor="white")
    ax.bar_label(bars, fmt="$%.0f", padding=4, fontsize=9)
    ax.set_ylabel("Revenue At Risk ($)")
    ax.set_title("Total Revenue At Risk by Product Category", fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    _save(fig, "13_revenue_at_risk_by_category.png")


# ---------------------------------------------
# 14. PAIR PLOT (NUMERIC FEATURES, SAMPLED)
# ---------------------------------------------
def plot_pairplot(df: pd.DataFrame):
    cols = ["Product_Price", "Order_Quantity", "User_Age",
            "Discount_Applied", "Return_Status"]
    sample = df[cols].sample(n=min(1500, len(df)), random_state=42)
    g = sns.pairplot(sample, hue="Return_Status",
                     palette={"Returned": DANGER, "Not Returned": SUCCESS},
                     plot_kws={"alpha": 0.4, "s": 15},
                     diag_kind="kde", corner=True)
    g.figure.suptitle("Pair Plot of Numeric Features by Return Status",
                       y=1.01, fontsize=13, fontweight="bold")
    _save(g.figure, "14_pairplot.png")


# ---------------------------------------------
# RUN ALL PLOTS
# ---------------------------------------------
def run_all_eda_plots(df: pd.DataFrame):
    print("\n[EDA] Generating visualizations...")
    plot_target_distribution(df)
    plot_return_by_category(df)
    plot_return_by_payment(df)
    plot_return_by_shipping(df)
    plot_return_reasons(df)
    plot_price_distribution(df)
    plot_age_distribution(df)
    plot_monthly_trend(df)
    plot_correlation_heatmap(df)
    plot_discount_vs_return(df)
    plot_quantity_vs_return(df)
    plot_gender_category_heatmap(df)
    plot_revenue_at_risk(df)
    plot_pairplot(df)
    print(f"[EDA] All 14 plots saved to -> {PLOT_DIR}/\n")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.s01_data_loading_and_quality import load_data
    from src.s02_data_cleaning_and_analytics import clean_data, calculate_sales
    df = calculate_sales(clean_data(load_data()))
    run_all_eda_plots(df)
