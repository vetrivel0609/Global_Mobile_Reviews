# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import pandas as pd
import numpy as np
import os

# ============================================================
# 2. LOAD DATA
# ============================================================
input_path = "data/Mobile Reviews Sentiment null.csv"
output_path = "outputs/cleaned_mobile_reviews.csv"

df = pd.read_csv(input_path)

print("=" * 60)
print("INITIAL DATASET")
print("=" * 60)
print("Shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())

# ============================================================
# 3. REMOVE DUPLICATES
# ============================================================
duplicate_count = df.duplicated().sum()

if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Duplicates removed:", duplicate_count)
else:
    print("No duplicate rows found.")

# ============================================================
# 4. CLEAN PRICE_LOCAL
# ============================================================
print("\nMissing price_local before cleaning:", df["price_local"].isna().sum())

df["price_local"] = (
    df["price_local"]
    .astype(str)
    .str.replace(r"[^\d.-]", "", regex=True)
    .replace("", np.nan)
)

df["price_local"] = pd.to_numeric(df["price_local"], errors="coerce")

print("Missing price_local after cleaning:", df["price_local"].isna().sum())

# ============================================================
# 5. FILL MISSING PRICE_USD
# ============================================================
missing_price_usd_before = df["price_usd"].isna().sum()

condition_usd = (
    df["price_usd"].isna()
    & df["price_local"].notna()
    & df["exchange_rate_to_usd"].notna()
)

df.loc[condition_usd, "price_usd"] = (
    df.loc[condition_usd, "price_local"]
    / df.loc[condition_usd, "exchange_rate_to_usd"]
)

filled_price_usd = condition_usd.sum()

print("Missing price_usd before filling:", missing_price_usd_before)
print("price_usd values filled:", filled_price_usd)
print("Missing price_usd after filling:", df["price_usd"].isna().sum())

# ============================================================
# 6. FILL MISSING PRICE_LOCAL
# ============================================================
missing_price_local_before = df["price_local"].isna().sum()

condition_local = (
    df["price_local"].isna()
    & df["price_usd"].notna()
    & df["exchange_rate_to_usd"].notna()
)

df.loc[condition_local, "price_local"] = (
    df.loc[condition_local, "price_usd"]
    * df.loc[condition_local, "exchange_rate_to_usd"]
)

filled_price_local = condition_local.sum()

print("Missing price_local before filling:", missing_price_local_before)
print("price_local values filled:", filled_price_local)
print("Missing price_local after filling:", df["price_local"].isna().sum())

# ============================================================
# 7. REMOVE ROWS WITH BOTH PRICES MISSING
# ============================================================
both_prices_missing = (
    df["price_usd"].isna()
    & df["price_local"].isna()
)

rows_removed_price = both_prices_missing.sum()

if rows_removed_price > 0:
    df = df.loc[~both_prices_missing].copy()

print("Rows removed because both prices were missing:", rows_removed_price)
print("Shape after price cleaning:", df.shape)

# ============================================================
# 8. CLEAN RATING
# ============================================================
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

print("Missing rating:", df["rating"].isna().sum())

# ============================================================
# 9. FILL MISSING RATING FROM SENTIMENT
# ============================================================
rating_from_sentiment = {
    "Negative": 2,
    "Neutral": 3,
    "Positive": 4
}

missing_rating = df["rating"].isna() & df["sentiment"].notna()

filled_rating = missing_rating.sum()

df.loc[missing_rating, "rating"] = (
    df.loc[missing_rating, "sentiment"]
    .map(rating_from_sentiment)
)

print("Rating values filled from sentiment:", filled_rating)
print("Missing rating after filling:", df["rating"].isna().sum())

# ============================================================
# 10. FILL MISSING SENTIMENT FROM RATING
# ============================================================
def rating_to_sentiment(rating):
    if rating in [1, 2]:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    elif rating in [4, 5]:
        return "Positive"
    return np.nan

missing_sentiment = (
    df["sentiment"].isna()
    & df["rating"].notna()
)

filled_sentiment = missing_sentiment.sum()

df.loc[missing_sentiment, "sentiment"] = (
    df.loc[missing_sentiment, "rating"]
    .apply(rating_to_sentiment)
)

print("Sentiment values filled from rating:", filled_sentiment)
print("Missing sentiment after filling:", df["sentiment"].isna().sum())

# ============================================================
# 11. REMOVE ROWS WITH BOTH RATING AND SENTIMENT MISSING
# ============================================================
both_rating_sentiment_missing = (
    df["rating"].isna()
    & df["sentiment"].isna()
)

rows_removed_rating = both_rating_sentiment_missing.sum()

if rows_removed_rating > 0:
    df = df.loc[~both_rating_sentiment_missing].copy()

print(
    "Rows removed because both rating and sentiment were missing:",
    rows_removed_rating
)

print("Shape after rating/sentiment cleaning:", df.shape)

# ============================================================
# 12. MAKE RATING AND SENTIMENT CONSISTENT
# ============================================================
df["sentiment"] = df["rating"].apply(rating_to_sentiment)

print("Rating and sentiment consistency completed.")

# ============================================================
# 13. CLEAN REVIEW_DATE
# ============================================================
df["review_date"] = pd.to_datetime(
    df["review_date"],
    errors="coerce"
)

print("Missing review_date:", df["review_date"].isna().sum())

# ============================================================
# 14. FILL MISSING SOURCE
# ============================================================
missing_source_before = df["source"].isna().sum()

if missing_source_before > 0:
    source_mode = df["source"].mode()[0]
    df["source"] = df["source"].fillna(source_mode)

print("Missing source before filling:", missing_source_before)
print("Missing source after filling:", df["source"].isna().sum())

# ============================================================
# 15. ROUND PRICE COLUMNS
# ============================================================
df["price_usd"] = df["price_usd"].round(2)
df["price_local"] = df["price_local"].round(2)

print("Price columns rounded to 2 decimal places.")

# ============================================================
# 16. FINAL CHECKS
# ============================================================
print("\n" + "=" * 60)
print("FINAL CLEANING CHECK")
print("=" * 60)

print("Final shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())

print("\nMissing values:")
print(df.isnull().sum())

print("\nRating distribution:")
print(df["rating"].value_counts().sort_index())

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("\nRating vs Sentiment:")
print(
    pd.crosstab(
        df["rating"],
        df["sentiment"]
    )
)

# ============================================================
# 17. SAVE CLEANED DATA
# ============================================================
os.makedirs("../outputs", exist_ok=True)

df.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("CLEANED DATA SAVED")
print("=" * 60)
print("File:", output_path)
print("Final shape:", df.shape)