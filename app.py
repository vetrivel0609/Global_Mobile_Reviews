# ============================================================
# IMPORT LIBRARIES
# ============================================================

import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Mobile Product ML System",
    page_icon="📱",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    cleaned = pd.read_csv(
        os.path.join(
            OUTPUT_DIR,
            "cleaned_mobile_reviews.csv"
        )
    )

    clustered = pd.read_csv(
        os.path.join(
            OUTPUT_DIR,
            "clustered_mobile_reviews.csv"
        )
    )

    cluster_analysis = pd.read_csv(
        os.path.join(
            OUTPUT_DIR,
            "cluster_analysis.csv"
        )
    )

    cluster_brand = pd.read_csv(
        os.path.join(
            OUTPUT_DIR,
            "cluster_brand_analysis.csv"
        )
    )

    cluster_model = pd.read_csv(
        os.path.join(
            OUTPUT_DIR,
            "cluster_model_analysis.csv"
        )
    )

    product_features = pd.read_csv(
        os.path.join(
            OUTPUT_DIR,
            "product_features.csv"
        )
    )

    product_similarity = pd.read_csv(
        os.path.join(
            OUTPUT_DIR,
            "product_similarity.csv"
        )
    )

    k_selection = pd.read_csv(
        os.path.join(
            OUTPUT_DIR,
            "k_selection_evaluation.csv"
        )
    )

    return (
        cleaned,
        clustered,
        cluster_analysis,
        cluster_brand,
        cluster_model,
        product_features,
        product_similarity,
        k_selection
    )


# ============================================================
# LOAD PROJECT DATA
# ============================================================

try:

    (
        cleaned,
        clustered,
        cluster_analysis,
        cluster_brand,
        cluster_model,
        product_features,
        product_similarity,
        k_selection
    ) = load_data()

except Exception as e:

    st.error("Unable to load project output files.")
    st.code(str(e))
    st.stop()


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("📱 Mobile Product ML System")

st.caption(
    "Product Segmentation • Cluster Analysis • "
    "Similarity-Based Recommendation"
)

st.divider()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "📈 EDA",
        "🎯 Segmentation",
        "💡 Insights & Reporting",
        "🤖 Recommendations",
        "📋 Evaluation"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "📊 Dashboard":

    st.header("Project Dashboard")

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Reviews",
        f"{len(cleaned):,}"
    )

    col2.metric(
        "Brands",
        cleaned["brand"].nunique()
    )

    col3.metric(
        "Products",
        cleaned["model"].nunique()
    )

    col4.metric(
        "Clusters",
        clustered["Cluster"].nunique()
    )

    st.divider()

    # --------------------------------------------------------
    # QUICK VISUALS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Reviews by Rating")

        rating_data = (
            cleaned["rating"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        rating_data.columns = [
            "Rating",
            "Reviews"
        ]

        fig = px.bar(
            rating_data,
            x="Rating",
            y="Reviews",
            text="Reviews",
            title="Rating Distribution"
        )

        fig.update_layout(
            height=350,
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("Reviews by Brand")

        brand_data = (
            cleaned["brand"]
            .value_counts()
            .reset_index()
        )

        brand_data.columns = [
            "Brand",
            "Reviews"
        ]

        fig = px.bar(
            brand_data,
            x="Brand",
            y="Reviews",
            text="Reviews",
            title="Brand Distribution"
        )

        fig.update_layout(
            height=350,
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.info(
        "The application uses the cleaned dataset, K-Means clustering "
        "results and the similarity-based recommendation model generated "
        "in the previous project stages."
    )


# ============================================================
# EDA
# ============================================================

elif page == "📈 EDA":

    st.header("📊 Exploratory Data Analysis")

    chart = st.selectbox(
        "Select Visualization",
        [
            "Average Rating by Brand",
            "Average Price by Brand",
            "Price vs Rating",
            "Sentiment Distribution",
            "Helpful Votes by Rating"
        ]
    )

    # --------------------------------------------------------
    # AVERAGE RATING BY BRAND
    # --------------------------------------------------------

    if chart == "Average Rating by Brand":

        data = (
            cleaned
            .groupby("brand")["rating"]
            .mean()
            .reset_index()
            .sort_values(
                "rating",
                ascending=False
            )
        )

        fig = px.bar(
            data,
            x="brand",
            y="rating",
            text_auto=".2f",
            title="Average Rating by Brand"
        )

        fig.update_yaxes(
            title="Average Rating",
            range=[0, 5]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # AVERAGE PRICE BY BRAND
    # --------------------------------------------------------

    elif chart == "Average Price by Brand":

        data = (
            cleaned
            .groupby("brand")["price_usd"]
            .mean()
            .reset_index()
            .sort_values(
                "price_usd",
                ascending=False
            )
        )

        fig = px.bar(
            data,
            x="brand",
            y="price_usd",
            text_auto=".2f",
            title="Average Price by Brand"
        )

        fig.update_yaxes(
            title="Average Price (USD)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # PRICE VS RATING
    # --------------------------------------------------------

    elif chart == "Price vs Rating":

        data = (
            cleaned
            .groupby(
                ["brand", "model"]
            )
            .agg(
                price_usd=("price_usd", "mean"),
                rating=("rating", "mean")
            )
            .reset_index()
        )

        fig = px.scatter(
            data,
            x="price_usd",
            y="rating",
            color="brand",
            hover_data=["model"],
            title="Product Price vs Average Rating"
        )

        fig.update_yaxes(
            range=[0, 5]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # SENTIMENT
    # --------------------------------------------------------

    elif chart == "Sentiment Distribution":

        data = (
            cleaned["sentiment"]
            .value_counts()
            .reset_index()
        )

        data.columns = [
            "Sentiment",
            "Reviews"
        ]

        fig = px.pie(
            data,
            names="Sentiment",
            values="Reviews",
            title="Sentiment Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # HELPFUL VOTES
    # --------------------------------------------------------

    elif chart == "Helpful Votes by Rating":

        data = (
            cleaned
            .groupby("rating")["helpful_votes"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            data,
            x="rating",
            y="helpful_votes",
            text_auto=".2f",
            title="Average Helpful Votes by Rating"
        )

        fig.update_xaxes(
            title="Rating"
        )

        fig.update_yaxes(
            title="Average Helpful Votes"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# SEGMENTATION
# ============================================================

elif page == "🎯 Segmentation":

    st.header("🎯 Product Segmentation")

    st.write(
        "K-Means groups reviews according to five "
        "product-performance rating features."
    )

    # --------------------------------------------------------
    # CLUSTER SUMMARY
    # --------------------------------------------------------

    cluster_summary = (
        clustered
        .groupby("Cluster")
        .agg(
            Reviews=("Cluster", "size"),
            Average_Rating=("rating", "mean"),
            Average_Price=("price_usd", "mean"),
            Battery=("battery_life_rating", "mean"),
            Camera=("camera_rating", "mean"),
            Performance=("performance_rating", "mean"),
            Design=("design_rating", "mean"),
            Display=("display_rating", "mean")
        )
        .reset_index()
    )

    cluster_summary = cluster_summary.round(2)

    st.subheader("Cluster Overview")

    st.dataframe(
        cluster_summary,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # CLUSTER SIZE
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            cluster_summary,
            names="Cluster",
            values="Reviews",
            title="Cluster Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            cluster_summary,
            x="Cluster",
            y="Average_Rating",
            text_auto=".2f",
            title="Average Rating by Cluster"
        )

        fig.update_yaxes(
            range=[0, 5]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # CLUSTER INSIGHTS
    # --------------------------------------------------------

    st.subheader("Cluster Insights")

    overall_average_rating = (
        cluster_summary["Average_Rating"].mean()
    )

    for _, row in cluster_summary.iterrows():

        cluster = int(row["Cluster"])

        if row["Average_Rating"] >= overall_average_rating:

            segment_name = "Higher-Rated Review Segment"

        else:

            segment_name = "Lower-Rated Review Segment"

        with st.expander(
            f"Cluster {cluster} — {segment_name}"
        ):

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Reviews",
                f"{int(row['Reviews']):,}"
            )

            col2.metric(
                "Average Rating",
                f"{row['Average_Rating']:.2f}"
            )

            col3.metric(
                "Average Price",
                f"${row['Average_Price']:.2f}"
            )

            st.write(
                f"""
                **Performance profile**

                - Battery: {row['Battery']:.2f}
                - Camera: {row['Camera']:.2f}
                - Performance: {row['Performance']:.2f}
                - Design: {row['Design']:.2f}
                - Display: {row['Display']:.2f}
                """
            )

    # --------------------------------------------------------
    # CLUSTER FEATURE PROFILE
    # --------------------------------------------------------

    st.subheader("Cluster Feature Profile")

    feature_columns = [
        "Battery",
        "Camera",
        "Performance",
        "Design",
        "Display"
    ]

    profile_data = (
        cluster_summary[
            ["Cluster"] + feature_columns
        ]
        .melt(
            id_vars="Cluster",
            var_name="Feature",
            value_name="Rating"
        )
    )

    fig = px.bar(
        profile_data,
        x="Feature",
        y="Rating",
        color="Cluster",
        barmode="group",
        title="Average Feature Ratings by Cluster"
    )

    fig.update_yaxes(
        range=[0, 5]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "The final clusters are performance/review-based segments. "
        "They should not be interpreted as premium, mid-range or budget "
        "segments because the cluster average prices are very similar."
    )


# ============================================================
# INSIGHTS & REPORTING
# ============================================================

elif page == "💡 Insights & Reporting":

    st.header("💡 Insights & Reporting")

    st.write(
        "Key findings from product segmentation, product performance, "
        "price-performance relationships, and customer preferences."
    )

    # ========================================================
    # 1. PRODUCT SEGMENTATION INSIGHTS
    # ========================================================

    st.subheader("🎯 Product Segmentation Insights")

    insight_cluster_summary = (
        clustered
        .groupby("Cluster")
        .agg(
            Reviews=("Cluster", "size"),
            Average_Price=("price_usd", "mean"),
            Average_Rating=("rating", "mean"),
            Battery=("battery_life_rating", "mean"),
            Camera=("camera_rating", "mean"),
            Performance=("performance_rating", "mean"),
            Design=("design_rating", "mean"),
            Display=("display_rating", "mean")
        )
        .round(2)
        .reset_index()
    )

    st.dataframe(
        insight_cluster_summary,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # CLUSTER PERFORMANCE INSIGHT
    # --------------------------------------------------------

    highest_cluster = insight_cluster_summary.loc[
        insight_cluster_summary["Average_Rating"].idxmax()
    ]

    lowest_cluster = insight_cluster_summary.loc[
        insight_cluster_summary["Average_Rating"].idxmin()
    ]

    col1, col2 = st.columns(2)

    with col1:

        st.success(
            f"""
            **Higher-Rated Review Segment**

            Cluster {int(highest_cluster["Cluster"])} has the higher
            average customer rating of
            **{highest_cluster["Average_Rating"]:.2f}**.

            Its component ratings are also higher across battery,
            camera, performance, design, and display.
            """
        )

    with col2:

        st.warning(
            f"""
            **Lower-Rated Review Segment**

            Cluster {int(lowest_cluster["Cluster"])} has the lower
            average customer rating of
            **{lowest_cluster["Average_Rating"]:.2f}**.

            Its component ratings are also lower across the selected
            product-performance features.
            """
        )

    # ========================================================
    # 2. HIGH-PERFORMING PRODUCTS
    # ========================================================

    st.subheader("🏆 High-Performing Products")

    product_performance = (
        clustered
        .groupby(
            ["brand", "model"]
        )
        .agg(
            Reviews=("review_id", "count"),
            Average_Rating=("rating", "mean"),
            Average_Price=("price_usd", "mean"),
            Performance=("performance_rating", "mean"),
            Camera=("camera_rating", "mean"),
            Battery=("battery_life_rating", "mean")
        )
        .reset_index()
    )

    product_performance = product_performance.round(2)

    high_products = (
        product_performance
        .sort_values(
            ["Average_Rating", "Reviews"],
            ascending=[False, False]
        )
        .head(10)
    )

    st.dataframe(
        high_products,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "High-performing products are identified using higher average "
        "customer ratings. Review count is also shown for context."
    )

    # ========================================================
    # 3. LOW-PERFORMING PRODUCTS
    # ========================================================

    st.subheader("⚠️ Low-Performing Products")

    low_products = (
        product_performance
        .sort_values(
            ["Average_Rating", "Reviews"],
            ascending=[True, False]
        )
        .head(10)
    )

    st.dataframe(
        low_products,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Low-performing products are identified using lower average "
        "customer ratings."
    )

    # ========================================================
    # 4. PRICE VS PERFORMANCE
    # ========================================================

    st.subheader("💰 Price vs Performance Trends")

    price_performance = (
        clustered
        .groupby("model")
        .agg(
            Average_Price=("price_usd", "mean"),
            Average_Rating=("rating", "mean"),
            Performance_Rating=("performance_rating", "mean")
        )
        .reset_index()
    )

    fig_price = px.scatter(
        price_performance,
        x="Average_Price",
        y="Average_Rating",
        size="Performance_Rating",
        hover_name="model",
        title="Price vs Customer Rating",
        labels={
            "Average_Price": "Average Price (USD)",
            "Average_Rating": "Average Customer Rating"
        }
    )

    fig_price.update_yaxes(
        range=[0, 5]
    )

    st.plotly_chart(
        fig_price,
        use_container_width=True
    )

    # --------------------------------------------------------
    # PRICE-RATING CORRELATION
    # --------------------------------------------------------

    price_rating_corr = (
        price_performance[
            [
                "Average_Price",
                "Average_Rating"
            ]
        ]
        .corr()
        .iloc[0, 1]
    )

    st.metric(
        "Price–Rating Correlation",
        f"{price_rating_corr:.2f}"
    )

    st.info(
        "The price-performance analysis helps examine whether higher "
        "priced products are associated with higher customer ratings."
    )

    # ========================================================
    # 5. CUSTOMER PREFERENCE PATTERNS
    # ========================================================

    st.subheader("👥 Customer Preference Patterns")

    col1, col2 = st.columns(2)

    with col1:

        rating_counts = (
            clustered["rating"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        rating_counts.columns = [
            "Rating",
            "Reviews"
        ]

        fig_rating = px.bar(
            rating_counts,
            x="Rating",
            y="Reviews",
            text="Reviews",
            title="Customer Ratings Distribution"
        )

        fig_rating.update_layout(
            height=400,
            showlegend=False
        )

        st.plotly_chart(
            fig_rating,
            use_container_width=True
        )

    with col2:

        brand_rating = (
            clustered
            .groupby("brand")["rating"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        brand_rating.columns = [
            "Brand",
            "Average Rating"
        ]

        fig_brand = px.bar(
            brand_rating,
            x="Brand",
            y="Average Rating",
            text_auto=".2f",
            title="Average Rating by Brand"
        )

        fig_brand.update_yaxes(
            range=[0, 5]
        )

        fig_brand.update_layout(
            height=400,
            showlegend=False
        )

        st.plotly_chart(
            fig_brand,
            use_container_width=True
        )

    # ========================================================
    # 6. CUSTOMER SENTIMENT
    # ========================================================

    st.subheader("😊 Customer Sentiment")

    sentiment_counts = (
        clustered["sentiment"]
        .value_counts()
        .reset_index()
    )

    sentiment_counts.columns = [
        "Sentiment",
        "Reviews"
    ]

    fig_sentiment = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Reviews",
        title="Customer Sentiment Distribution"
    )

    st.plotly_chart(
        fig_sentiment,
        use_container_width=True
    )

    # ========================================================
    # 7. DATA-DRIVEN DECISION SUPPORT
    # ========================================================

    st.subheader("📌 Data-Driven Decision Support")

    st.markdown(
        """
        ### Key Business Insights

        **1. Product Segmentation**

        K-Means clustering identifies two major review-performance
        segments based on battery, camera, performance, design, and
        display ratings.

        **2. Product Performance**

        Products can be compared using customer ratings and product
        component ratings to identify higher- and lower-performing
        products.

        **3. Price vs Performance**

        Price and customer rating can be compared to understand whether
        higher-priced products are associated with better customer
        ratings.

        **4. Customer Preferences**

        Rating, sentiment, brand, model, and component-level ratings
        provide information about customer preferences.

        **5. Recommendation Support**

        The recommendation system uses product characteristics and
        cosine similarity to identify products with similar feature
        profiles.

        **6. Business Decision Making**

        These insights can support product comparison, market
        segmentation, product positioning, and recommendation-based
        decisions.
        """
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    st.success(
        "Insights & Reporting completed successfully using the "
        "project's cleaned data, clustering results, EDA findings, "
        "and recommendation features."
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

elif page == "🤖 Recommendations":

    st.header("🤖 Product Recommendations")

    st.write(
        "Select a mobile product to find products with similar "
        "characteristics."
    )

    products = sorted(
        product_features["Product_Name"]
        .dropna()
        .unique()
    )

    selected_product = st.selectbox(
        "Select Product",
        products
    )

    top_n = st.slider(
        "Number of Recommendations",
        1,
        10,
        5
    )

    if st.button(
        "🔎 Generate Recommendations",
        type="primary"
    ):

        recommendations = product_similarity[
            product_similarity["Product_Name"] == selected_product
        ].copy()

        recommendations = (
            recommendations
            .sort_values(
                "Similarity_Score",
                ascending=False
            )
            .head(top_n)
        )

        if recommendations.empty:

            st.warning(
                "No recommendations available for this product."
            )

        else:

            st.subheader(
                f"Similar Products to {selected_product}"
            )

            display_data = recommendations[
                [
                    "Recommended_Product",
                    "Similarity_Score"
                ]
            ].copy()

            display_data["Similarity_Score"] = (
                display_data["Similarity_Score"]
                .round(4)
            )

            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True
            )
            # ------------------------------------------------
            # RECOMMENDATION CHART
            # ------------------------------------------------
            fig = px.bar(
                display_data,
                x="Similarity_Score",
                y="Recommended_Product",
                orientation="h",
                text_auto=".3f",
                title="Recommendation Similarity"
            )

            fig.update_layout(
                yaxis={
                    "categoryorder": "total ascending"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.divider()

    st.subheader("Recommendation Method")

    st.write(
        """
        The recommendation system uses:

        **Product Features → Standard Scaling → "
        "Cosine Similarity → Top-N Recommendations**
        """
    )

    st.caption(
        "Features include price, overall rating and component ratings."
    )

# ============================================================
# MODEL EVALUATION
# ============================================================
elif page == "📋 Evaluation":

    st.header("📈 Model Evaluation")

    # --------------------------------------------------------
    # FINAL MODEL METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Silhouette",
        "0.4032"
    )

    col2.metric(
        "Davies-Bouldin",
        "0.9689"
    )

    col3.metric(
        "Calinski-Harabasz",
        "48,650.90"
    )

    col4.metric(
        "K",
        "2"
    )

    st.divider()
    # --------------------------------------------------------
    # K SELECTION
    # --------------------------------------------------------
    st.subheader("Cluster Count Selection")

    fig = px.line(
        k_selection,
        x="K",
        y="Silhouette_Score",
        markers=True,
        title="Silhouette Score by Number of Clusters"
    )

    fig.update_yaxes(
        title="Silhouette Score"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "K = 2 produced the highest Silhouette Score among the tested "
        "cluster counts from K = 2 to K = 8."
    )
    # --------------------------------------------------------
    # COMPLETE K SELECTION RESULTS
    # --------------------------------------------------------
    with st.expander(
        "View Complete K Selection Results"
    ):

        st.dataframe(
            k_selection,
            use_container_width=True,
            hide_index=True
        )

# ============================================================
# SIDEBAR FOOTER
# ============================================================
st.sidebar.divider()

st.sidebar.caption(
    "Mobile Product Segmentation & Recommendation System"
)

st.sidebar.caption(
    "Python • Pandas • Scikit-learn • Plotly • Streamlit"
)