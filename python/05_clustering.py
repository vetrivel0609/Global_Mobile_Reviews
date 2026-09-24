# ============================================================
# 05. K-MEANS CLUSTERING
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "outputs", "cleaned_mobile_reviews.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "clustered_mobile_reviews.csv")
CHARTS_PATH = os.path.join(BASE_DIR, "outputs", "charts")

os.makedirs(CHARTS_PATH, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

print("=" * 60)
print("K-MEANS CLUSTERING")
print("=" * 60)

print("\nInput Shape:", df.shape)

cluster_features = [
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating"
]

X = df[cluster_features].copy()

print("\nClustering Features:")
print(cluster_features)

print("\nClustering Data Shape:", X.shape)
print("Missing Values:", X.isna().sum().sum())

# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled Data Shape:", X_scaled.shape)

# ============================================================
# K-MEANS MODEL
# ============================================================

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nK-Means clustering completed.")

# ============================================================
# CLUSTER DISTRIBUTION
# ============================================================

cluster_counts = df["Cluster"].value_counts().sort_index()

print("\nCluster Distribution:")
print(cluster_counts)

cluster_percentage = (
    df["Cluster"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

print("\nCluster Percentage:")
print(cluster_percentage)

# ============================================================
# CLUSTER PROFILE
# ============================================================

cluster_profile = (
    df.groupby("Cluster")
    .agg(
        Records=("model", "count"),
        Unique_Models=("model", "nunique"),
        Unique_Brands=("brand", "nunique"),
        Average_Price=("price_usd", "mean"),
        Average_Rating=("rating", "mean"),
        Battery_Rating=("battery_life_rating", "mean"),
        Camera_Rating=("camera_rating", "mean"),
        Performance_Rating=("performance_rating", "mean"),
        Design_Rating=("design_rating", "mean"),
        Display_Rating=("display_rating", "mean"),
        Average_Helpful_Votes=("helpful_votes", "mean")
    )
    .round(2)
)

print("\nCluster Profile:")
print(cluster_profile)

# ============================================================
# PRICE BY CLUSTER
# ============================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="Cluster",
    y="price_usd",
    estimator="mean",
    errorbar=None
)

plt.title("Average Price by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Price (USD)")
plt.tight_layout()

plt.savefig(
    os.path.join(CHARTS_PATH, "average_price_by_cluster.png"),
    dpi=300
)

plt.close()

# ============================================================
# RATING BY CLUSTER
# ============================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="Cluster",
    y="rating",
    estimator="mean",
    errorbar=None
)

plt.title("Average Rating by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Rating")
plt.tight_layout()

plt.savefig(
    os.path.join(CHARTS_PATH, "average_rating_by_cluster.png"),
    dpi=300
)

plt.close()

# ============================================================
# CLUSTER FEATURE PROFILE
# ============================================================

cluster_means = (
    df.groupby("Cluster")[cluster_features]
    .mean()
)

plt.figure(figsize=(10, 6))

sns.heatmap(
    cluster_means,
    annot=True,
    fmt=".2f"
)

plt.title("Cluster Feature Profile")
plt.xlabel("Features")
plt.ylabel("Cluster")
plt.tight_layout()

plt.savefig(
    os.path.join(CHARTS_PATH, "cluster_profile_heatmap.png"),
    dpi=300
)

plt.close()

# ============================================================
# SAVE CLUSTERED DATA
# ============================================================

df.to_csv(OUTPUT_PATH, index=False)

print("\nClustered file saved:")
print(OUTPUT_PATH)

print("\nClustering completed successfully.")