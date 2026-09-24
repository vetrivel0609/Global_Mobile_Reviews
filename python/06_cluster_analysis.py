# ============================================================
# IMPORTS LIBRARIES
# ============================================================
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "cluster_analysis.csv"
)
BRAND_OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "cluster_brand_analysis.csv"
)
MODEL_OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "cluster_model_analysis.csv"
)
CHARTS_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "charts"
)
os.makedirs(CHARTS_PATH, exist_ok=True)

# ============================================================
# 2. LOAD CLUSTERED DATA
# ============================================================
df = pd.read_csv(INPUT_PATH)

print("=" * 70)
print("CLUSTER ANALYSIS")
print("=" * 70)

print("\nInput Shape:", df.shape)

print("\nAvailable Clusters:")
print(sorted(df["Cluster"].unique()))

# ============================================================
# 3. CLUSTER DISTRIBUTION
# ============================================================
cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)
cluster_percentage = (
    df["Cluster"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)
cluster_distribution = pd.DataFrame({
    "Records": cluster_counts,
    "Percentage": cluster_percentage
})
print("\n" + "=" * 70)
print("CLUSTER DISTRIBUTION")
print("=" * 70)

print(cluster_distribution)

# ============================================================
# 4. DETAILED CLUSTER PROFILE
# ============================================================
cluster_profile = (
    df.groupby("Cluster")
    .agg(
        Records=("review_id", "count"),
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
cluster_profile["Percentage"] = (
    cluster_profile["Records"]
    / len(df)
    * 100
).round(2)

print("\n" + "=" * 70)
print("DETAILED CLUSTER PROFILE")
print("=" * 70)

print(cluster_profile)

# ============================================================
# 5. IDENTIFY CLUSTER CHARACTERISTICS
# ============================================================
feature_columns = [
    "Average_Price",
    "Average_Rating",
    "Battery_Rating",
    "Camera_Rating",
    "Performance_Rating",
    "Design_Rating",
    "Display_Rating"
]
feature_means = cluster_profile[feature_columns].mean()

print("\n" + "=" * 70)
print("CLUSTER CHARACTERISTICS")
print("=" * 70)

for cluster in cluster_profile.index:

    print(f"\nCluster {cluster}:")

    price = cluster_profile.loc[
        cluster, "Average_Price"
    ]
    rating = cluster_profile.loc[
        cluster, "Average_Rating"
    ]
    battery = cluster_profile.loc[
        cluster, "Battery_Rating"
    ]
    camera = cluster_profile.loc[
        cluster, "Camera_Rating"
    ]
    performance = cluster_profile.loc[
        cluster, "Performance_Rating"
    ]
    design = cluster_profile.loc[
        cluster, "Design_Rating"
    ]
    display = cluster_profile.loc[
        cluster, "Display_Rating"
    ]
    print(f"  Average Price       : ${price:.2f}")
    print(f"  Average Rating      : {rating:.2f}")
    print(f"  Battery Rating      : {battery:.2f}")
    print(f"  Camera Rating       : {camera:.2f}")
    print(f"  Performance Rating  : {performance:.2f}")
    print(f"  Design Rating       : {design:.2f}")
    print(f"  Display Rating      : {display:.2f}")

# ============================================================
# 6. BRAND DISTRIBUTION BY CLUSTER
# ============================================================
cluster_brand = (
    df.groupby(["Cluster", "brand"])
    .size()
    .reset_index(name="Count")
)
cluster_brand["Percentage"] = (
    cluster_brand.groupby("Cluster")["Count"]
    .transform(lambda x: x / x.sum() * 100)
).round(2)
cluster_brand = cluster_brand.sort_values(
    ["Cluster", "Count"],
    ascending=[True, False]
)
print("\n" + "=" * 70)
print("TOP BRANDS IN EACH CLUSTER")
print("=" * 70)

for cluster in sorted(df["Cluster"].unique()):

    print(f"\nCluster {cluster}:")

    temp = (
        cluster_brand[
            cluster_brand["Cluster"] == cluster
        ]
        .head(5)
    )
    print(
        temp[
            ["brand", "Count", "Percentage"]
        ].to_string(index=False)
    )

# ============================================================
# 7. MODEL DISTRIBUTION BY CLUSTER
# ============================================================
cluster_model = (
    df.groupby(["Cluster", "model"])
    .size()
    .reset_index(name="Count")
)
cluster_model["Percentage"] = (
    cluster_model.groupby("Cluster")["Count"]
    .transform(lambda x: x / x.sum() * 100)
).round(2)
cluster_model = cluster_model.sort_values(
    ["Cluster", "Count"],
    ascending=[True, False]
)
print("\n" + "=" * 70)
print("TOP MODELS IN EACH CLUSTER")
print("=" * 70)

for cluster in sorted(df["Cluster"].unique()):

    print(f"\nCluster {cluster}:")

    temp = (
        cluster_model[
            cluster_model["Cluster"] == cluster
        ]
        .head(5)
    )
    print(
        temp[
            ["model", "Count", "Percentage"]
        ].to_string(index=False)
    )

# ============================================================
# 8. HIGH AND LOW PRICE CLUSTERS
# ============================================================
highest_price_cluster = cluster_profile[
    "Average_Price"
].idxmax()

lowest_price_cluster = cluster_profile[
    "Average_Price"
].idxmin()

highest_rating_cluster = cluster_profile[
    "Average_Rating"
].idxmax()

lowest_rating_cluster = cluster_profile[
    "Average_Rating"
].idxmin()

print("\n" + "=" * 70)
print("KEY CLUSTER COMPARISONS")
print("=" * 70)

print(
    f"\nHighest average price: "
    f"Cluster {highest_price_cluster}"
)

print(
    f"Price: "
    f"${cluster_profile.loc[highest_price_cluster, 'Average_Price']:.2f}"
)

print(
    f"\nLowest average price: "
    f"Cluster {lowest_price_cluster}"
)

print(
    f"Price: "
    f"${cluster_profile.loc[lowest_price_cluster, 'Average_Price']:.2f}"
)

print(
    f"\nHighest average rating: "
    f"Cluster {highest_rating_cluster}"
)

print(
    f"Rating: "
    f"{cluster_profile.loc[highest_rating_cluster, 'Average_Rating']:.2f}"
)

print(
    f"\nLowest average rating: "
    f"Cluster {lowest_rating_cluster}"
)

print(
    f"Rating: "
    f"{cluster_profile.loc[lowest_rating_cluster, 'Average_Rating']:.2f}"
)

# ============================================================
# 9. PRICE VS RATING BY CLUSTER
# ============================================================
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="price_usd",
    y="rating",
    hue="Cluster",
    palette="Set2",
    alpha=0.5
)
plt.title("Cluster Analysis: Price vs Rating")
plt.xlabel("Price (USD)")
plt.ylabel("Rating")
plt.legend(title="Cluster")
plt.tight_layout()

plt.savefig(
    os.path.join(
        CHARTS_PATH,
        "cluster_analysis_price_rating.png"
    ),
    dpi=300
)
plt.show()
plt.close()

# ============================================================
# 10. AVERAGE PRICE COMPARISON
# ============================================================
plt.figure(figsize=(8, 5))

sns.barplot(
    data=cluster_profile.reset_index(),
    x="Cluster",
    y="Average_Price"
)
plt.title("Average Price by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Price (USD)")
plt.tight_layout()

plt.savefig(
    os.path.join(
        CHARTS_PATH,
        "cluster_analysis_average_price.png"
    ),
    dpi=300
)
plt.show()
plt.close()

# ============================================================
# 11. AVERAGE RATING COMPARISON
# ============================================================
plt.figure(figsize=(8, 5))

sns.barplot(
    data=cluster_profile.reset_index(),
    x="Cluster",
    y="Average_Rating"
)
plt.title("Average Rating by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Rating")
plt.tight_layout()

plt.savefig(
    os.path.join(
        CHARTS_PATH,
        "cluster_analysis_average_rating.png"
    ),
    dpi=300
)
plt.show()
plt.close()

# ============================================================
# 12. PRODUCT QUALITY COMPARISON
# ============================================================
quality_columns = [
    "Battery_Rating",
    "Camera_Rating",
    "Performance_Rating",
    "Design_Rating",
    "Display_Rating"
]
quality_data = cluster_profile[
    quality_columns
].reset_index()

quality_long = quality_data.melt(
    id_vars="Cluster",
    var_name="Feature",
    value_name="Average_Rating"
)
plt.figure(figsize=(12, 6))

sns.barplot(
    data=quality_long,
    x="Feature",
    y="Average_Rating",
    hue="Cluster"
)
plt.title("Product Quality Features by Cluster")
plt.xlabel("Feature")
plt.ylabel("Average Rating")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    os.path.join(
        CHARTS_PATH,
        "cluster_quality_comparison.png"
    ),
    dpi=300
)
plt.show()
plt.close()

# ============================================================
# 13. CLUSTER PROFILE HEATMAP
# ============================================================
heatmap_columns = [
    "Average_Price",
    "Average_Rating",
    "Battery_Rating",
    "Camera_Rating",
    "Performance_Rating",
    "Design_Rating",
    "Display_Rating"
]
plt.figure(figsize=(12, 6))

sns.heatmap(
    cluster_profile[heatmap_columns],
    annot=True,
    fmt=".2f"
)
plt.title("Detailed Cluster Profile")
plt.xlabel("Features")
plt.ylabel("Cluster")
plt.tight_layout()

plt.savefig(
    os.path.join(
        CHARTS_PATH,
        "detailed_cluster_profile_heatmap.png"
    ),
    dpi=300
)
plt.show()
plt.close()

# ============================================================
# 14. SAVE CLUSTER PROFILE
# ============================================================
cluster_profile.to_csv(
    OUTPUT_PATH
)
cluster_brand.to_csv(
    BRAND_OUTPUT_PATH,
    index=False
)
cluster_model.to_csv(
    MODEL_OUTPUT_PATH,
    index=False
)

# ============================================================
# 15. FINAL OUTPUT
# ============================================================
print("\n" + "=" * 70)
print("CLUSTER ANALYSIS COMPLETED")
print("=" * 70)

print("\nCluster Analysis saved:")
print(OUTPUT_PATH)

print("\nBrand Analysis saved:")
print(BRAND_OUTPUT_PATH)

print("\nModel Analysis saved:")
print(MODEL_OUTPUT_PATH)

print("\nCharts saved in:")
print(CHARTS_PATH)