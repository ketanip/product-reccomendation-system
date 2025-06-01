import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import joblib
import os

# Constants
DATA_PATH = "data.csv"
FILTERED_CSV = "filtered_data.csv"
REDUCED_FEATURES = "X_reduced.npy"
PREPROCESSOR_PATH = "preprocessor.pkl"
SVD_COMPONENTS = 50  # Reduce this if needed to minimize model size further

# Load data
df = pd.read_csv(DATA_PATH)

# Drop unwanted column
if "Product_Size" in df.columns:
    df = df.drop("Product_Size", axis=1)

# Features
numeric_features = ["Price_USD", "Rating", "Number_of_Reviews"]
categorical_features = ["Brand", "Category", "Usage_Frequency", "Skin_Type",
                        "Gender_Target", "Packaging_Type", "Main_Ingredient",
                        "Cruelty_Free", "Country_of_Origin"]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ]
)

# Apply preprocessing
X = preprocessor.fit_transform(df)

# Dimensionality Reduction
svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=42)
X_reduced = svd.fit_transform(X)

# Save reduced data and filtered DataFrame
np.save(REDUCED_FEATURES, X_reduced)
df.to_csv(FILTERED_CSV, index=False)

# Optionally save preprocessor and SVD model
joblib.dump(preprocessor, PREPROCESSOR_PATH)
joblib.dump(svd, "svd_model.pkl")

print("✅ Preprocessing complete.")
print(f"➡️ Saved: {FILTERED_CSV}, {REDUCED_FEATURES}, {PREPROCESSOR_PATH}")
