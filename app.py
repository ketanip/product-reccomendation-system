import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load data and model artifacts
df = pd.read_csv("filtered_data.csv")
X_reduced = np.load("X_reduced.npy")
product_names = df["Product_Name"]

# Streamlit UI
st.set_page_config(page_title="🛍️ Product Recommender", layout="wide")
st.title("🛍️ Product Recommendation System")

# Sidebar filters
st.sidebar.header("🔍 Filter Products")

min_price = float(df["Price_USD"].min())
max_price = float(df["Price_USD"].max())

selected_brand = st.sidebar.selectbox("Brand", ["All"] + sorted(df["Brand"].unique()))
selected_skin = st.sidebar.selectbox("Skin Type", ["All"] + sorted(df["Skin_Type"].unique()))
selected_gender = st.sidebar.selectbox("Gender Target", ["All"] + sorted(df["Gender_Target"].unique()))
selected_cruelty = st.sidebar.selectbox("Cruelty-Free", ["All", True, False])
price_range = st.sidebar.slider("💰 Price Range", min_value=min_price,
                                max_value=max_price,
                                value=(min_price, max_price))

# Apply filters
filtered_df = df.copy()
if selected_brand != "All":
    filtered_df = filtered_df[filtered_df["Brand"] == selected_brand]
if selected_skin != "All":
    filtered_df = filtered_df[filtered_df["Skin_Type"] == selected_skin]
if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["Gender_Target"] == selected_gender]
if selected_cruelty != "All":
    filtered_df = filtered_df[filtered_df["Cruelty_Free"] == selected_cruelty]

filtered_df = filtered_df[(filtered_df["Price_USD"] >= price_range[0]) &
                          (filtered_df["Price_USD"] <= price_range[1])]

filtered_product_names = filtered_df["Product_Name"].tolist()

# Recommendation function
def recommend(product_name, top_n=3):
    if product_name not in product_names.values:
        return pd.DataFrame()
    idx = product_names[product_names == product_name].index[0]
    similarities = cosine_similarity(X_reduced[idx].reshape(1, -1), X_reduced)[0]
    sim_scores = list(enumerate(similarities))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    indices = [i[0] for i in sim_scores]
    return df.iloc[indices]

# Product selector
selected_product = st.selectbox("Choose a product to get recommendations:", filtered_product_names)

# Show recommendations
if st.button("Get Recommendations"):
    recommendations = recommend(selected_product, top_n=3)

    if not recommendations.empty:
        st.subheader("You might also like:")

        for _, row in recommendations.iterrows():
            st.markdown("---")

            # Search URLs
            query = f"{row['Product_Name']} {row['Brand']}".replace(" ", "+")
            amazon_url = f"https://www.amazon.in/s?k={query}"
            flipkart_url = f"https://www.flipkart.com/search?q={query}"

            # Display product in table format
            st.markdown(f"""
            <table style='width:100%; border:1px solid #ddd; border-collapse: collapse; font-size:16px;'>
                <tr><th style='text-align:left; padding:6px;'>🏷️ Product Name</th><td style='padding:6px;'>{row['Product_Name']}</td></tr>
                <tr><th style='text-align:left; padding:6px;'>Brand</th><td style='padding:6px;'>{row['Brand']}</td></tr>
                <tr><th style='text-align:left; padding:6px;'>Price</th><td style='padding:6px;'>${row['Price_USD']}</td></tr>
                <tr><th style='text-align:left; padding:6px;'>Rating</th><td style='padding:6px;'>⭐ {row['Rating']} ({row['Number_of_Reviews']} reviews)</td></tr>
                <tr><th style='text-align:left; padding:6px;'>Skin Type</th><td style='padding:6px;'>{row['Skin_Type']}</td></tr>
                <tr><th style='text-align:left; padding:6px;'>Gender Target</th><td style='padding:6px;'>{row['Gender_Target']}</td></tr>
                <tr><th style='text-align:left; padding:6px;'>Packaging</th><td style='padding:6px;'>{row['Packaging_Type']}</td></tr>
                <tr><th style='text-align:left; padding:6px;'>Main Ingredient</th><td style='padding:6px;'>{row['Main_Ingredient']}</td></tr>
                <tr><th style='text-align:left; padding:6px;'>Cruelty-Free</th><td style='padding:6px;'>{'✅ Yes' if row['Cruelty_Free'] else '❌ No'}</td></tr>
                <tr><th style='text-align:left; padding:6px;'>Buy Online</th>
                    <td style='padding:6px;'>
                        <a href='{amazon_url}' target='_blank'>
                            <button style='background-color:#FF9900; color:white; border:none; padding:6px 12px; margin-right:10px;'>🛒 Amazon</button>
                        </a>
                        <a href='{flipkart_url}' target='_blank'>
                            <button style='background-color:#2874F0; color:white; border:none; padding:6px 12px;'>🛍️ Flipkart</button>
                        </a>
                    </td>
                </tr>
            </table>
            """, unsafe_allow_html=True)
    else:
        st.warning("No recommendations found.")
