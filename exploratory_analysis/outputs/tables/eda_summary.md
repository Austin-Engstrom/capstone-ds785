# Exploratory Data Analysis Summary

## Dataset Overview
- Reviews: 297
- Columns: 25
- Unique Brands: 151
- Product Categories: 3
- Review Types: 6
- Date Range: 2017-11-29 22:38:00+00:00 to 2026-07-06 15:01:00+00:00

## Article Length
- Mean Word Count: 2241.3
- Median Word Count: 1972.0
- Minimum Word Count: 200
- Maximum Word Count: 5611
- Mean Average Word Length: 4.41

## Retail Price
- Reviews with Price: 279 of 297
- Missing Price Percentage: 6.1%
- Mean Retail Price: $3,429
- Median Retail Price: $1,250
- Maximum Retail Price: $14,999
- Retail Price Skewness: 0.83

## Missing Values
- Product Name Missing Percentage: 13.1%
- Retail Price Missing Percentage: 6.1%
- Article Text Missing Percentage: 0.0%

## Key Findings
- The dataset contains long-form professional mountain bike product reviews rather than short customer reviews.
- Retail price is right-skewed, which is expected because premium bicycles and components can have very high MSRPs.
- Brand representation is spread across many manufacturers, reducing the risk that the model only learns patterns from one dominant brand.
- Product categories are somewhat imbalanced, with fewer clothing reviews than component and protective gear reviews.
- Product name extraction remains a future ETL improvement because many professional review articles do not expose a single clean product name field.
- Engineered text features such as word count, title length, publication year, and price availability provide useful context for downstream sentiment modeling.