# ============================================================
# IMPORTS LIBRARIES
# ============================================================
import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 1. PROJECT PATHS
# ============================================================
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
INPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "clustered_mobile_reviews.csv"
)
PRODUCT_OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "product_features.csv"
)
SIMILARITY_OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "product_similarity.csv"
)

# ============================================================
# 2. LOAD DATA
# ============================================================
df = pd.read_csv(INPUT_PATH)

print("=" * 70)
print("MOBILE PRODUCT RECOMMENDATION SYSTEM")
print("=" * 70)

print("\nInput Shape:", df.shape)

# ============================================================
# 3. SELECT PRODUCT FEATURES
# ============================================================
product_features = [
    "price_usd",
    "rating",
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating"
]
print("\nRecommendation Features:")
print(product_features)

# ============================================================
# 4. CREATE PRODUCT-LEVEL DATA
# ============================================================
product_df = (
    df.groupby(["brand", "model"], as_index=False)[product_features]
    .mean()
)
print("\nProduct-Level Shape:", product_df.shape)

print("\nNumber of Unique Products:")
print(product_df["model"].nunique())

# ============================================================
# 5. CREATE PRODUCT NAME
# ============================================================
product_df["Product_Name"] = (
    product_df["brand"] + " " + product_df["model"]
)
print("\nProduct Examples:")
print(
    product_df[
        ["Product_Name"] + product_features
    ].head(10)
)

# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================
print("\nMissing Values in Recommendation Features:")

missing = product_df[product_features].isna().sum()

print(missing)

if missing.sum() == 0:
    print("No missing values found.")
else:
    print("Missing values detected.")

# ============================================================
# 7. STANDARDIZE PRODUCT FEATURES
# ============================================================
scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    product_df[product_features]
)

print("\nScaled Product Feature Shape:")
print(X_scaled.shape)

# ============================================================
# 8. CALCULATE COSINE SIMILARITY
# ============================================================
similarity_matrix = cosine_similarity(X_scaled)

print("\nSimilarity Matrix Shape:")
print(similarity_matrix.shape)

# ============================================================
# 9. CREATE SIMILARITY DATAFRAME
# ============================================================
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=product_df["Product_Name"],
    columns=product_df["Product_Name"]
)

# ============================================================
# 10. RECOMMENDATION FUNCTION
# ============================================================
def recommend_products(product_name, top_n=5):

    if product_name not in similarity_df.index:
        print("\nProduct not found.")
        print("\nAvailable Products:")

        for product in product_df["Product_Name"]:
            print("-", product)

        return pd.DataFrame()

    similarity_scores = (
        similarity_df[product_name]
        .sort_values(
            ascending=False
        )
    )

    # Remove the selected product itself
    similarity_scores = similarity_scores.drop(
        product_name
    )

    recommendations = (
        similarity_scores
        .head(top_n)
        .reset_index()
    )

    recommendations.columns = [
        "Product_Name",
        "Similarity_Score"
    ]

    recommendations["Similarity_Score"] = (
        recommendations["Similarity_Score"]
        .round(4)
    )

    return recommendations

# ============================================================
# 11. DISPLAY ALL PRODUCTS
# ============================================================
print("\n" + "=" * 70)
print("AVAILABLE PRODUCTS")
print("=" * 70)

for product in product_df["Product_Name"]:
    print(product)

# ============================================================
# 12. TEST RECOMMENDATION
# ============================================================
test_product = product_df["Product_Name"].iloc[0]

print("\n" + "=" * 70)
print("TEST RECOMMENDATION")
print("=" * 70)

print("\nSelected Product:")
print(test_product)

recommendations = recommend_products(
    test_product,
    top_n=5
)

print("\nRecommended Products:")

if not recommendations.empty:
    print(
        recommendations.to_string(
            index=False
        )
    )

# ============================================================
# 13. SAVE PRODUCT FEATURES
# ============================================================

product_df.to_csv(
    PRODUCT_OUTPUT_PATH,
    index=False
)
print("\nProduct feature dataset saved:")
print(PRODUCT_OUTPUT_PATH)

# ============================================================
# 14. CREATE LONG-FORM SIMILARITY DATA
# ============================================================
similarity_records = []

for product in similarity_df.index:

    scores = (
        similarity_df[product]
        .drop(product)
        .sort_values(
            ascending=False
        )
    )

    for recommended_product, score in scores.items():

        similarity_records.append({
            "Product_Name": product,
            "Recommended_Product": recommended_product,
            "Similarity_Score": round(score, 4)
        })


similarity_output = pd.DataFrame(
    similarity_records
)

# ============================================================
# 15. SAVE SIMILARITY DATA
# ============================================================
similarity_output.to_csv(
    SIMILARITY_OUTPUT_PATH,
    index=False
)
print("\nSimilarity dataset saved:")
print(SIMILARITY_OUTPUT_PATH)

# ============================================================
# 16. FINAL CHECK
# ============================================================
print("\n" + "=" * 70)
print("RECOMMENDATION SYSTEM COMPLETED")
print("=" * 70)

print("\nNumber of Products:")
print(len(product_df))

print("\nNumber of Similarity Records:")
print(len(similarity_output))

print("\nRecommendation Features:")
print(product_features)