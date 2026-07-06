"""
EDA summary script for DS 785 Presentation 3.

Generates:
- dataset overview
- missing value summary
- descriptive statistics
- distribution statistics
- outlier summary
- category / review type / brand counts
- engineered text features
- visualizations for presentation
- markdown EDA summary
"""

from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt


# File paths

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "processed" / "pinkbike_reviews.csv"

OUTPUT_DIR = BASE_DIR / "exploratory_analysis" / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# Load data

df = pd.read_csv(INPUT_FILE)

df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce", utc=True)
df["modified_date"] = pd.to_datetime(df["modified_date"], errors="coerce", utc=True)


# Feature engineering for EDA

df["article_word_count"] = (
    df["article_text"]
    .fillna("")
    .str.split()
    .str.len()
)

df["title_length"] = (
    df["title"]
    .fillna("")
    .str.len()
)

df["title_word_count"] = (
    df["title"]
    .fillna("")
    .str.split()
    .str.len()
)

df["avg_word_length"] = (
    df["article_text"]
    .fillna("")
    .apply(
        lambda text: (
            sum(len(word) for word in re.findall(r"\w+", text))
            / len(re.findall(r"\w+", text))
        )
        if re.findall(r"\w+", text)
        else 0
    )
)

df["publish_year"] = df["publish_date"].dt.year
df["publish_month"] = df["publish_date"].dt.to_period("M").astype(str)

df["has_price"] = df["retail_price"].notna()
df["has_product_name"] = df["product_name"].notna()


# Dataset overview

overview = pd.DataFrame(
    {
        "metric": [
            "rows",
            "columns",
            "unique_brands",
            "unique_categories",
            "unique_review_types",
            "date_min",
            "date_max",
            "avg_article_words",
            "median_article_words",
            "avg_word_length",
            "reviews_with_price",
            "reviews_with_product_name",
        ],
        "value": [
            len(df),
            len(df.columns),
            df["brand"].nunique(),
            df["product_category"].nunique(),
            df["review_type"].nunique(),
            df["publish_date"].min(),
            df["publish_date"].max(),
            round(df["article_word_count"].mean(), 2),
            round(df["article_word_count"].median(), 2),
            round(df["avg_word_length"].mean(), 2),
            int(df["has_price"].sum()),
            int(df["has_product_name"].sum()),
        ],
    }
)

overview.to_csv(TABLE_DIR / "dataset_overview.csv", index=False)


# Missing values

missing_values = (
    df.isna()
    .sum()
    .reset_index()
    .rename(columns={"index": "column", 0: "missing_count"})
)

missing_values["missing_pct"] = (
    missing_values["missing_count"] / len(df) * 100
).round(2)

missing_values = missing_values.sort_values("missing_count", ascending=False)

missing_values.to_csv(TABLE_DIR / "missing_values.csv", index=False)


# Descriptive statistics

numeric_cols = [
    "retail_price",
    "article_text_length",
    "article_word_count",
    "title_length",
    "title_word_count",
    "avg_word_length",
]

descriptive_stats = df[numeric_cols].describe().round(2)
descriptive_stats.to_csv(TABLE_DIR / "descriptive_statistics.csv")


# Distribution statistics

distribution_stats = pd.DataFrame(
    {
        "mean": df[numeric_cols].mean(),
        "median": df[numeric_cols].median(),
        "std_dev": df[numeric_cols].std(),
        "skewness": df[numeric_cols].skew(),
        "kurtosis": df[numeric_cols].kurtosis(),
    }
).round(2)

distribution_stats.to_csv(TABLE_DIR / "distribution_statistics.csv")


# Outlier summary using IQR method

def iqr_outlier_count(series):
    series = series.dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return ((series < lower_bound) | (series > upper_bound)).sum()


outlier_summary = pd.DataFrame(
    {
        "variable": numeric_cols,
        "outlier_count": [
            iqr_outlier_count(df[col]) for col in numeric_cols
        ],
    }
)

outlier_summary["outlier_pct"] = (
    outlier_summary["outlier_count"] / len(df) * 100
).round(2)

outlier_summary.to_csv(TABLE_DIR / "outlier_summary.csv", index=False)


# Frequency tables

df["brand"].value_counts().to_csv(TABLE_DIR / "brand_counts.csv")
df["product_category"].value_counts().to_csv(TABLE_DIR / "category_counts.csv")
df["product_subcategory"].value_counts().to_csv(TABLE_DIR / "subcategory_counts.csv")
df["review_type"].value_counts().to_csv(TABLE_DIR / "review_type_counts.csv")
df["publish_year"].value_counts().sort_index().to_csv(TABLE_DIR / "publish_year_counts.csv")
df["publish_month"].value_counts().sort_index().to_csv(TABLE_DIR / "publish_month_counts.csv")


# Visualizations

def save_bar_chart(series, title, xlabel, ylabel, filename, top_n=None):
    if top_n:
        series = series.head(top_n)

    plt.figure(figsize=(10, 6))
    series.sort_values().plot(kind="barh")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=300)
    plt.close()


# Article length histogram

plt.figure(figsize=(10, 6))
df["article_word_count"].plot(kind="hist", bins=30)
plt.title("Distribution of Article Word Counts")
plt.xlabel("Word Count")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "article_word_count_histogram.png", dpi=300)
plt.close()


# Retail price histogram

plt.figure(figsize=(10, 6))
df["retail_price"].dropna().plot(kind="hist", bins=30)
plt.title("Distribution of Retail Prices")
plt.xlabel("Retail Price")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "retail_price_histogram.png", dpi=300)
plt.close()


# Missing values chart

missing_plot = missing_values[missing_values["missing_count"] > 0].sort_values(
    "missing_count",
    ascending=True,
)

plt.figure(figsize=(8, 6))
plt.barh(missing_plot["column"], missing_plot["missing_count"])
plt.title("Missing Values by Variable")
plt.xlabel("Missing Records")
plt.ylabel("Column")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "missing_values.png", dpi=300)
plt.close()


# Top brands

save_bar_chart(
    df["brand"].value_counts(),
    title="Top 20 Brands by Review Count",
    xlabel="Number of Reviews",
    ylabel="Brand",
    filename="top_20_brands.png",
    top_n=20,
)


# Product categories

save_bar_chart(
    df["product_category"].value_counts(),
    title="Product Category Distribution",
    xlabel="Number of Reviews",
    ylabel="Category",
    filename="product_category_counts.png",
)


# Review types

save_bar_chart(
    df["review_type"].value_counts(),
    title="Review Type Distribution",
    xlabel="Number of Reviews",
    ylabel="Review Type",
    filename="review_type_counts.png",
)


# Publication year trend

plt.figure(figsize=(10, 6))
df["publish_year"].value_counts().sort_index().plot(kind="bar")
plt.title("Reviews by Publication Year")
plt.xlabel("Publication Year")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "reviews_by_year.png", dpi=300)
plt.close()


# Publication month timeline

timeline = df["publish_month"].value_counts().sort_index()

plt.figure(figsize=(12, 5))
timeline.plot(kind="line")
plt.title("Reviews Published Over Time")
plt.xlabel("Publication Month")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "publication_timeline.png", dpi=300)
plt.close()


# Correlation heatmap

corr = df[numeric_cols].corr(numeric_only=True)

plt.figure(figsize=(8, 6))
plt.imshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar(label="Correlation")
plt.title("Correlation Matrix of Numeric Features")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "numeric_correlation_matrix.png", dpi=300)
plt.close()


# Markdown summary

summary = f"""
# Exploratory Data Analysis Summary

## Dataset Overview
- Reviews: {len(df)}
- Columns: {len(df.columns)}
- Unique Brands: {df["brand"].nunique()}
- Product Categories: {df["product_category"].nunique()}
- Review Types: {df["review_type"].nunique()}
- Date Range: {df["publish_date"].min()} to {df["publish_date"].max()}

## Article Length
- Mean Word Count: {df["article_word_count"].mean():.1f}
- Median Word Count: {df["article_word_count"].median():.1f}
- Minimum Word Count: {df["article_word_count"].min():.0f}
- Maximum Word Count: {df["article_word_count"].max():.0f}
- Mean Average Word Length: {df["avg_word_length"].mean():.2f}

## Retail Price
- Reviews with Price: {df["has_price"].sum()} of {len(df)}
- Missing Price Percentage: {df["retail_price"].isna().mean() * 100:.1f}%
- Mean Retail Price: ${df["retail_price"].mean():,.0f}
- Median Retail Price: ${df["retail_price"].median():,.0f}
- Maximum Retail Price: ${df["retail_price"].max():,.0f}
- Retail Price Skewness: {df["retail_price"].skew():.2f}

## Missing Values
- Product Name Missing Percentage: {df["product_name"].isna().mean() * 100:.1f}%
- Retail Price Missing Percentage: {df["retail_price"].isna().mean() * 100:.1f}%
- Article Text Missing Percentage: {df["article_text"].isna().mean() * 100:.1f}%

## Key Findings
- The dataset contains long-form professional mountain bike product reviews rather than short customer reviews.
- Retail price is right-skewed, which is expected because premium bicycles and components can have very high MSRPs.
- Brand representation is spread across many manufacturers, reducing the risk that the model only learns patterns from one dominant brand.
- Product categories are somewhat imbalanced, with fewer clothing reviews than component and protective gear reviews.
- Product name extraction remains a future ETL improvement because many professional review articles do not expose a single clean product name field.
- Engineered text features such as word count, title length, publication year, and price availability provide useful context for downstream sentiment modeling.
"""

with open(TABLE_DIR / "eda_summary.md", "w") as f:
    f.write(summary.strip())


# Save enriched EDA dataset

df.to_csv(TABLE_DIR / "pinkbike_reviews_with_eda_features.csv", index=False)

print("EDA complete.")
print(f"Tables saved to: {TABLE_DIR}")
print(f"Figures saved to: {FIGURE_DIR}")