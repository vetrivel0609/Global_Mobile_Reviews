# ============================================================
# Import Libraries and Define Paths
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "clustered_mobile_reviews.csv"
)

SIMILARITY_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "product_similarity.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "model_evaluation.csv"
)

K_EVALUATION_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "k_selection_evaluation.csv"
)

CHARTS_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "charts"
)

os.makedirs(CHARTS_PATH, exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_PATH)

similarity_df = pd.read_csv(
    SIMILARITY_PATH
)

print("=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print("\nClustering Data Shape:", df.shape)

print(
    "Similarity Data Shape:",
    similarity_df.shape
)

# ============================================================
# 2. SELECT FINAL CLUSTERING FEATURES
# ============================================================

cluster_features = [
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating"
]

X = df[cluster_features].copy()

print("\nFinal Clustering Features:")
print(cluster_features)

# ============================================================
# 3. STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled Feature Shape:")
print(X_scaled.shape)

# ============================================================
# 4. EXISTING FINAL CLUSTERS
# ============================================================

existing_clusters = df["Cluster"]

number_of_clusters = existing_clusters.nunique()

print("\nExisting Number of Clusters:")
print(number_of_clusters)

# ============================================================
# 5. CLUSTER EVALUATION
# ============================================================

silhouette = silhouette_score(
    X_scaled,
    existing_clusters
)

davies_bouldin = davies_bouldin_score(
    X_scaled,
    existing_clusters
)

calinski_harabasz = calinski_harabasz_score(
    X_scaled,
    existing_clusters
)

kmeans_final = KMeans(
    n_clusters=number_of_clusters,
    random_state=42,
    n_init=10
)

kmeans_final.fit(X_scaled)

inertia = kmeans_final.inertia_

print("\n" + "=" * 70)
print("CLUSTER EVALUATION RESULTS")
print("=" * 70)

print(
    f"\nSilhouette Score       : {silhouette:.4f}"
)

print(
    f"Davies-Bouldin Index  : {davies_bouldin:.4f}"
)

print(
    f"Calinski-Harabasz     : {calinski_harabasz:.4f}"
)

print(
    f"Inertia                : {inertia:.2f}"
)

# ============================================================
# 6. TEST DIFFERENT K VALUES
# ============================================================

print("\n" + "=" * 70)
print("K SELECTION EVALUATION")
print("=" * 70)

k_results = []

for k in range(2, 9):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(
        X_scaled
    )

    silhouette_k = silhouette_score(
        X_scaled,
        labels
    )

    davies_bouldin_k = davies_bouldin_score(
        X_scaled,
        labels
    )

    calinski_harabasz_k = (
        calinski_harabasz_score(
            X_scaled,
            labels
        )
    )

    inertia_k = kmeans.inertia_

    k_results.append({
        "K": k,
        "Silhouette_Score": round(
            silhouette_k,
            4
        ),
        "Davies_Bouldin_Index": round(
            davies_bouldin_k,
            4
        ),
        "Calinski_Harabasz_Index": round(
            calinski_harabasz_k,
            2
        ),
        "Inertia": round(
            inertia_k,
            2
        )
    })

k_results_df = pd.DataFrame(
    k_results
)

print(
    k_results_df.to_string(
        index=False
    )
)

# ============================================================
# 7. IDENTIFY BEST SILHOUETTE K
# ============================================================

best_k_silhouette = (
    k_results_df
    .loc[
        k_results_df[
            "Silhouette_Score"
        ].idxmax(),
        "K"
    ]
)

best_silhouette_value = (
    k_results_df[
        "Silhouette_Score"
    ].max()
)

print(
    f"\nHighest Silhouette Score K: "
    f"{int(best_k_silhouette)}"
)

print(
    f"Highest Silhouette Score: "
    f"{best_silhouette_value:.4f}"
)

# ============================================================
# 8. ELBOW CHART
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    k_results_df["K"],
    k_results_df["Inertia"],
    marker="o"
)

plt.title("K-Means Elbow Curve")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(
    k_results_df["K"]
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHARTS_PATH,
        "kmeans_elbow_curve.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# 9. SILHOUETTE SCORE CHART
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    k_results_df["K"],
    k_results_df["Silhouette_Score"],
    marker="o"
)

plt.title(
    "Silhouette Score by Number of Clusters"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")

plt.xticks(
    k_results_df["K"]
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHARTS_PATH,
        "silhouette_score_by_k.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# 10. DAVIES-BOULDIN CHART
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    k_results_df["K"],
    k_results_df["Davies_Bouldin_Index"],
    marker="o"
)

plt.title(
    "Davies-Bouldin Index by K"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Davies-Bouldin Index")

plt.xticks(
    k_results_df["K"]
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHARTS_PATH,
        "davies_bouldin_by_k.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# 11. RECOMMENDATION EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("RECOMMENDATION EVALUATION")
print("=" * 70)

recommendation_count = (
    len(similarity_df)
)

average_similarity = (
    similarity_df[
        "Similarity_Score"
    ].mean()
)

median_similarity = (
    similarity_df[
        "Similarity_Score"
    ].median()
)

minimum_similarity = (
    similarity_df[
        "Similarity_Score"
    ].min()
)

maximum_similarity = (
    similarity_df[
        "Similarity_Score"
    ].max()
)

print(
    f"\nTotal Similarity Comparisons: "
    f"{recommendation_count}"
)

print(
    f"Average Similarity Score: "
    f"{average_similarity:.4f}"
)

print(
    f"Median Similarity Score: "
    f"{median_similarity:.4f}"
)

print(
    f"Minimum Similarity Score: "
    f"{minimum_similarity:.4f}"
)

print(
    f"Maximum Similarity Score: "
    f"{maximum_similarity:.4f}"
)

# ============================================================
# 12. TOP RECOMMENDATION SIMILARITY
# ============================================================

top_recommendations = (
    similarity_df
    .sort_values(
        "Similarity_Score",
        ascending=False
    )
    .groupby("Product_Name")
    .head(5)
)

average_top_similarity = (
    top_recommendations[
        "Similarity_Score"
    ].mean()
)

print(
    f"\nAverage Top-5 Similarity Score: "
    f"{average_top_similarity:.4f}"
)

# ============================================================
# 13. SIMILARITY SCORE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    similarity_df["Similarity_Score"],
    bins=20
)

plt.title(
    "Distribution of Product Similarity Scores"
)

plt.xlabel(
    "Cosine Similarity Score"
)

plt.ylabel(
    "Number of Comparisons"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHARTS_PATH,
        "similarity_score_distribution.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# 14. SAVE K EVALUATION
# ============================================================

k_results_df.to_csv(
    K_EVALUATION_PATH,
    index=False
)

# ============================================================
# 15. SAVE MODEL EVALUATION SUMMARY
# ============================================================

evaluation_summary = pd.DataFrame({
    "Metric": [
        "Number_of_Clusters",
        "Clustering_Features",
        "Silhouette_Score",
        "Davies_Bouldin_Index",
        "Calinski_Harabasz_Index",
        "Inertia",
        "Best_K_by_Silhouette",
        "Best_Silhouette_Score",
        "Total_Similarity_Comparisons",
        "Average_Similarity_Score",
        "Median_Similarity_Score",
        "Minimum_Similarity_Score",
        "Maximum_Similarity_Score",
        "Average_Top5_Similarity"
    ],

    "Value": [
        number_of_clusters,
        ", ".join(cluster_features),
        round(silhouette, 4),
        round(davies_bouldin, 4),
        round(calinski_harabasz, 2),
        round(inertia, 2),
        int(best_k_silhouette),
        round(best_silhouette_value, 4),
        recommendation_count,
        round(average_similarity, 4),
        round(median_similarity, 4),
        round(minimum_similarity, 4),
        round(maximum_similarity, 4),
        round(average_top_similarity, 4)
    ]
})

evaluation_summary.to_csv(
    OUTPUT_PATH,
    index=False
)

# ============================================================
# 16. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION COMPLETED")
print("=" * 70)

print("\nEvaluation summary saved:")
print(OUTPUT_PATH)

print("\nK-selection results saved:")
print(K_EVALUATION_PATH)

print("\nEvaluation charts saved in:")
print(CHARTS_PATH)