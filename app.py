import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
from PIL import Image
from streamlit_option_menu import option_menu
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)
def load_css():

    with open("assets/style.css") as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()
with st.sidebar:

    selected = option_menu(

        menu_title="Shopper Spectrum",

        options=[

            "Home",

            "Dashboard",

            "Customer Segmentation",

            "Recommendation",

            "Dataset",

            "About"

        ],

        icons=[

            "house",

            "bar-chart",

            "people",

            "cart",

            "table",

            "info-circle"

        ],

        menu_icon="shop",

        default_index=0,
    )
df = pd.read_csv("data/cleaned_online_retail.csv")

segments = pd.read_csv("data/customer_segments.csv")

similarity = joblib.load("models/product_similarity.pkl")

kmeans = joblib.load("models/kmeans_model.pkl")

scaler = joblib.load("models/scaler.pkl")

if selected == "Home":

    st.markdown("""

    # 🛒 Shopper Spectrum

    ### Customer Segmentation & Product Recommendation

    """)
    col1,col2,col3,col4 = st.columns(4)

    with col1:

        st.metric(
            "Customers",
            df["CustomerID"].nunique()
        )

    with col2:

        st.metric(
            "Products",
            df["Description"].nunique()
        )

    with col3:

        st.metric(
            "Revenue",
            f"${df['TotalPrice'].sum():,.0f}"
        )

    with col4:

        st.metric(
            "Countries",
            df["Country"].nunique()
        )

    st.markdown("""

    ## Business Overview

    Explore customer trends, sales performance, and product demand across the retail dataset.

    """)

    top_revenue = df.groupby("Country")["TotalPrice"].sum().reset_index().sort_values(by="TotalPrice", ascending=False).head(7)
    top_products = df.groupby("Description")["Quantity"].sum().reset_index().sort_values(by="Quantity", ascending=False).head(10)

    fig1 = px.bar(top_revenue, x="Country", y="TotalPrice", title="Top Revenue by Country", labels={"TotalPrice":"Revenue"})
    fig2 = px.bar(top_products, x="Quantity", y="Description", orientation="h", title="Top 10 Products by Quantity", labels={"Quantity":"Units Sold"})

    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

elif selected == "Dashboard":

    st.markdown("""

    # 📊 Dashboard

    Comprehensive business analytics for customer segments, sales, and product performance.

    """)

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    revenue_by_country = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head(8).reset_index()
    orders_by_month = df.groupby(df["InvoiceDate"].dt.to_period("M"))["InvoiceNo"].nunique().reset_index()
    orders_by_month["InvoiceDate"] = orders_by_month["InvoiceDate"].dt.to_timestamp()
    segment_counts = segments["Segment"].value_counts().reset_index()
    segment_counts.columns = ["Segment", "Count"]

    r1, r2, r3 = st.columns(3)
    r1.metric("Total Customers", df["CustomerID"].nunique())
    r2.metric("Total Orders", df["InvoiceNo"].nunique())
    r3.metric("Average Order Value", f"${df['TotalPrice'].mean():,.2f}")

    fig_country = px.bar(revenue_by_country, x="Country", y="TotalPrice", title="Revenue by Country", labels={"TotalPrice":"Revenue"})
    fig_orders = px.line(orders_by_month, x="InvoiceDate", y="InvoiceNo", title="Orders by Month", labels={"InvoiceNo":"Unique Orders"})
    fig_segment = px.pie(segment_counts, names="Segment", values="Count", title="Customer Segment Distribution")

    st.plotly_chart(fig_country, use_container_width=True)
    st.plotly_chart(fig_orders, use_container_width=True)
    st.plotly_chart(fig_segment, use_container_width=True)

elif selected == "Customer Segmentation":

    st.markdown("""

    # 👥 Customer Segmentation

    Predict customer segments from Recency, Frequency, and Monetary values.

    """)

    with st.form("segmentation_form"):

        recency = st.number_input("Recency (days)", min_value=0, max_value=1000, value=int(segments["Recency"].median()))
        frequency = st.number_input("Frequency (transactions)", min_value=0, max_value=1000, value=int(segments["Frequency"].median()))
        monetary = st.number_input("Monetary (spend)", min_value=0.0, max_value=1000000.0, value=float(segments["Monetary"].median()), format="%.2f")
        submitted = st.form_submit_button("Predict Segment")

    if submitted:
        input_data = scaler.transform([[recency, frequency, monetary]])
        cluster_id = int(kmeans.predict(input_data)[0])
        segment_map = segments.groupby("Cluster")["Segment"].agg(lambda x: x.mode().iat[0]).to_dict()
        segment_label = segment_map.get(cluster_id, "Unknown")

        st.success(f"Predicted Segment: {segment_label}")
        st.write(f"Cluster: {cluster_id}")
        st.write("Use these values to identify where a customer falls in your retention and loyalty strategy.")

    st.markdown("""

    ### RFM Inputs

    Enter Recency, Frequency and Monetary values to predict the customer segment. These values are scaled and classified using the trained clustering model.

    """)

elif selected == "Recommendation":

    st.markdown("""

    # 🛍️ Recommendation

    Find the most similar products based on item similarity.

    """)

    product_list = similarity.index.tolist()
    selected_product = st.selectbox("Select a product", product_list)
    recommend_button = st.button("Recommend Products")

    if recommend_button:
        if selected_product in similarity.index:
            nearest = similarity.loc[selected_product].sort_values(ascending=False).drop(labels=[selected_product]).head(10)
            rec_df = pd.DataFrame({"Recommended Product": nearest.index, "Similarity": nearest.values})
            st.dataframe(rec_df)
        else:
            st.warning("The selected product is not available in the similarity model.")

elif selected == "Dataset":

    st.markdown("""

    # 📁 Dataset

    Explore and download the cleaned retail dataset.

    """)

    search_term = st.text_input("Search by product description or country")
    country_options = sorted(df["Country"].unique())
    selected_countries = st.multiselect("Filter by country", country_options, default=country_options[:5])

    filtered_df = df.copy()
    if search_term:
        filtered_df = filtered_df[filtered_df["Description"].str.contains(search_term, case=False, na=False) | filtered_df["Country"].str.contains(search_term, case=False, na=False)]
    if selected_countries:
        filtered_df = filtered_df[filtered_df["Country"].isin(selected_countries)]

    st.markdown(f"**Showing {len(filtered_df):,} records**")
    st.dataframe(filtered_df.reset_index(drop=True))

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered dataset", data=csv_data, file_name="filtered_online_retail.csv", mime="text/csv")

elif selected == "About":

    st.markdown("""

    # ℹ️ About

    Shopper Spectrum is a customer segmentation and product recommendation system built with Streamlit.

    """)

    st.markdown("""

    - Uses RFM analysis with a trained clustering model for customer segmentation.
    - Includes item-to-item product recommendations through a similarity matrix.
    - Displays commercial analytics, dataset exploration, and model-driven predictions.

    """)

    st.markdown("""

    ### Loaded Dataset and Models

    - `data/cleaned_online_retail.csv`
    - `data/customer_segments.csv`
    - `models/kmeans_model.pkl`
    - `models/scaler.pkl`
    - `models/product_similarity.pkl`

    """)
