# ============================================================
# 05. PRODUCT-LEVEL CLUSTERING EXPERIMENT
# ============================================================

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "outputs", "cleaned_mobile_reviews.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "product_clustering_experiment.csv")

df = pd.read_csv(INPUT_PATH)

print("=" * 60)
print("PRODUCT-LEVEL CLUSTERING EXPERIMENT")
print("=" * 60)

print("\nReview-Level Shape:", df.shape)
print("Unique Products:", df[["brand", "model"]].drop_duplicates().shape[0])

# ============================================================
# PRODUCT-LEVEL AGGREGATION
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

product_df = (
    df.groupby(["brand", "model"])[product_features]
    .mean()
    .reset_index()
)

print("\nProduct-Level Shape:", product_df.shape)

print("\nProduct-Level Data:")
print(product_df.head())

# ============================================================
# FEATURE SETS
# ============================================================

feature_sets = {
    "Current_7_Features": [
        "price_usd",
        "rating",
        "battery_life_rating",
        "camera_rating",
        "performance_rating",
        "design_rating",
        "display_rating"
    ],

    "Performance_Features": [
        "battery_life_rating",
        "camera_rating",
        "performance_rating",
        "design_rating",
        "display_rating"
    ],

    "Price_and_Performance": [
        "price_usd",
        "performance_rating",
        "battery_life_rating",
        "camera_rating"
    ]
}

results = []

# ============================================================
# K-MEANS EXPERIMENT
# ============================================================

for feature_name, features in feature_sets.items():

    print("\n" + "=" * 60)
    print("FEATURE SET:", feature_name)
    print("=" * 60)

    X = product_df[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for k in range(2, 9):

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(X_scaled)

        silhouette = silhouette_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)
        calinski = calinski_harabasz_score(X_scaled, labels)

        results.append({
            "Feature_Set": feature_name,
            "K": k,
            "Silhouette_Score": round(silhouette, 4),
            "Davies_Bouldin_Index": round(davies_bouldin, 4),
            "Calinski_Harabasz_Index": round(calinski, 2)
        })

        print(
            f"K={k} | "
            f"Silhouette={silhouette:.4f} | "
            f"DB={davies_bouldin:.4f} | "
            f"CH={calinski:.2f}"
        )

# ============================================================
# BEST RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("BEST PRODUCT-LEVEL RESULTS")
print("=" * 60)

best_results = (
    results_df
    .sort_values("Silhouette_Score", ascending=False)
    .head(10)
)

print(best_results.to_string(index=False))

# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(OUTPUT_PATH, index=False)

print("\nExperiment results saved:")
print(OUTPUT_PATH)