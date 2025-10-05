import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
import numpy as np
import joblib
import logging

logging.basicConfig(
    filename="logs/training.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def train_model():
    # Load data
    df = pd.read_csv("data/processed/features.csv")

    # Define target
    y = 21 - df["finishing_position"]

    # Features we keep
    X = df.drop(columns=["finishing_position", "driver_name"])  

    # Categorical and numeric splits
    categorical_features = ["circuit", "constructor", "dominant_wind_dir", "race"]
    numeric_features = [col for col in X.columns if col not in categorical_features and col != "date"]

    # Extract season/year/month from date
    X["year"] = pd.to_datetime(df["date"]).dt.year
    X["month"] = pd.to_datetime(df["date"]).dt.month
    X = X.drop(columns=["date"])

    race_ids = df["race_id"].unique()
    train_ids, test_ids = train_test_split(race_ids, test_size=0.2, random_state=42)

    train_mask = df["race_id"].isin(train_ids)
    test_mask = df["race_id"].isin(test_ids)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # Groups (counts of drivers per race) for train/test
    group_train = df[train_mask].groupby("race_id").size().to_numpy()
    group_test = df[test_mask].groupby("race_id").size().to_numpy()

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features + ["year", "month"])
        ]
    )

    # XGBoost ranker
    ranker = xgb.XGBRanker(
        objective="rank:ndcg",
        learning_rate=0.05,
        max_depth=6,
        n_estimators=200,
        random_state=42,
    )

    # Full pipeline
    pipeline = make_pipeline(preprocessor, ranker)

    # Fit model (must include group info)
    pipeline.fit(X_train, y_train, xgbranker__group=group_train)

    y_pred = pipeline.predict(X_test)

    # Evaluate with NDCG@10
    # We need to reshape per-race
    ndcgs = []
    winner_correct = 0
    total_races = 0

    for race in df[test_mask]["race_id"].unique():
        mask = df[test_mask]["race_id"] == race
        true = y_test[mask].values.reshape(1, -1)
        pred = y_pred[mask].reshape(1, -1)

        # NDCG@10
        ndcgs.append(ndcg_score(true, pred, k=10))

        # Winner accuracy
        actual_winner_idx = np.argmax(true)
        predicted_winner_idx = np.argmax(pred)
        if actual_winner_idx == predicted_winner_idx:
            winner_correct += 1
        total_races += 1

    logging.info(f"Avg NDCG@10: {np.mean(ndcgs):.4f}")
    logging.info(f"Winner accuracy: {winner_correct}/{total_races} = {winner_correct/total_races:.2%}")

    # Save model
    joblib.dump(pipeline, "f1_ranker_model.pkl")
    logging.info("Model trained and saved as f1_ranker_model.pkl")
