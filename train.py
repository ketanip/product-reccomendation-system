import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "data.csv"
MODEL_PATH = "recommendation_model.pkl"

# Load data
df = pd.read_csv(DATA_PATH)
df = df.drop("Product_Size", axis=1)

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

if __name__ == "__main__":
    build_and_save_model(df)
    print(f"Model saved to {MODEL_PATH}")
