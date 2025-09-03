import pandas as pd
import joblib
from src.data.build_dataset import build_latest_race_dataset

def run_prediction():
    # Load the trained pipeline
    pipeline = joblib.load("f1_ranker_model.pk1")

    # Load new data (must match training features)
    new_df = build_latest_race_dataset()

    # Prepare features (same preprocessing as training)
    X_new = new_df.drop(columns=["driver_name", "dnf"])
    X_new["year"] = pd.to_datetime(new_df["date"]).dt.year
    X_new["month"] = pd.to_datetime(new_df["date"]).dt.month
    X_new = X_new.drop(columns=["date"])

    # Predict scores
    scores = pipeline.predict(X_new)
    new_df["predicted_score"] = scores
    new_df["predicted_rank"] = new_df.groupby("race")["predicted_score"] \
                                    .rank(method="first", ascending=False)

    results = new_df[["race", "driver_name", "driver_number", "predicted_rank"]] \
              .sort_values(["race", "predicted_rank"])

    print("Predicted Finishing Order:")
    results.to_csv("data/predictions/predictions_log.csv", index=False)
    print(results)

    return results
