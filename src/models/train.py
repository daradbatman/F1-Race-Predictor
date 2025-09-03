import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import joblib

def train_model():
    # Load data
    df = pd.read_csv("data/processed/features.csv")

    # Define target
    y = df["finishing_position"]

    # Features we keep
    X = df.drop(columns=["finishing_position", "driver_name", "dnf"])  

    # Categorical and numeric splits
    categorical_features = ["circuit", "constructor", "dominant_wind_dir", "race"]
    numeric_features = [col for col in X.columns if col not in categorical_features and col != "date"]

    # Extract season/year/month from date
    X["year"] = pd.to_datetime(df["date"]).dt.year
    X["month"] = pd.to_datetime(df["date"]).dt.month
    X = X.drop(columns=["date"])

    group = df.groupby("race").size().to_numpy()

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features + ["year", "month"])
        ]
    )

    # XGBoost model
    ranker = xgb.XGBRanker(
        objective="rank:ndcg",
        learning_rate=0.05,
        max_depth=6,
        n_estimators=200,
        random_state=42,
    )

    # Full pipeline
    pipeline = make_pipeline(preprocessor, ranker)

    # 4. Fit model (must include group info)
    pipeline.fit(X, y, xgbranker__group=group)

    joblib.dump(pipeline, "f1_ranker_model.pk1")
    print("Model trained and saved as f1_ranker_model.pk1")

