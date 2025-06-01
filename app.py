import os
import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "data.csv"
MODEL_PATH = "recommendation_model.pkl"

# Load data
df = pd.read_csv(DATA_PATH)
df = df.drop("Product_Size", axis=1)

# Price range setup
min_price = float(df["Price_USD"].min())
max_price = float(df["Price_USD"].max())

# Features
numeric_features = ["Price_USD", "Rating", "Number_of_Reviews"]
categorical_features = ["Brand", "Category", "Usage_Frequency", "Skin_Type",
                        "Gender_Target", "Packaging_Type", "Main_Ingredient",
                        "Cruelty_Free", "Country_of_Origin"]

def build_and_save_model(df):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )
    X = preprocessor.fit_transform(df)
    similarity = cosine_similarity(X)
    product_names = df["Product_Name"].values

    model_data = {
        "preprocessor": preprocessor,
        "similarity": similarity,
        "product_names": product_names
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    
    return model_data

# Load or build model
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
else:
    model_data = build_and_save_model(df)

preprocessor = model_data["preprocessor"]
similarity = model_data["similarity"]
product_names = model_data["product_names"]

# Recommendation function
def recommend(product_name, top_n=3):
    if product_name not in product_names:
        return pd.DataFrame()
    idx = list(product_names).index(product_name)
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    indices = [i[0] for i in sim_scores]
    return df.iloc[indices]

# Streamlit UI
st.title("🛍️ Product Recommendation System")

# Sidebar filters
st.sidebar.header("🔍 Filter Products")

selected_brand = st.sidebar.selectbox("Brand", ["All"] + sorted(df["Brand"].unique()))
selected_skin = st.sidebar.selectbox("Skin Type", ["All"] + sorted(df["Skin_Type"].unique()))
selected_gender = st.sidebar.selectbox("Gender Target", ["All"] + sorted(df["Gender_Target"].unique()))
selected_cruelty = st.sidebar.selectbox("Cruelty-Free", ["All", True, False])
price_range = st.sidebar.slider("💰 Price Range", min_value=float(min_price),
                                max_value=float(max_price),
                                value=(float(min_price), float(max_price)))

# Filter products
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

# Product selector
selected_product = st.selectbox("Choose a product to get recommendations:", filtered_product_names)

# Show recommendations
if st.button("Get Recommendations"):
    recommendations = recommend(selected_product, top_n=3)

    if not recommendations.empty:
        st.subheader("You might also like:")

        for _, row in recommendations.iterrows():
            st.markdown("---")

            # Generate search URLs
            query = f"{row['Product_Name']} {row['Brand']}".replace(" ", "+")
            amazon_url = f"https://www.amazon.in/s?k={query}"
            flipkart_url = f"https://www.flipkart.com/search?q={query}"

            # Build HTML table with embedded buttons
            st.markdown(f"""
            <table style='width:100%; border:1px solid #ddd; border-collapse: collapse;'>
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
