import pandas as pd
import joblib
from src.data.build_dataset import build_latest_race_dataset
import logging

logging.basicConfig(
    filename="logs/prediction.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def run_prediction():
    # Load the trained pipeline
    pipeline = joblib.load("f1_ranker_model.pkl")

    # Load new data (must match training features)
    new_df = build_latest_race_dataset()
    output_path = f"data/predictions/prediction_log.csv"

    # Prepare features (same preprocessing as training)
    if new_df is not None and not new_df.empty: 
        X_new = new_df.drop(columns=["driver_name"])
        X_new["year"] = pd.to_datetime(new_df["date"]).dt.year
        X_new["month"] = pd.to_datetime(new_df["date"]).dt.month
        X_new = X_new.drop(columns=["date"])

        # Predict scores
        scores = pipeline.predict(X_new)
        new_df["predicted_score"] = scores
        new_df["predicted_rank"] = new_df.groupby("race_id")["predicted_score"] \
                                        .rank(method="first", ascending=False)

        results = new_df[["race_id", "race", "driver_name", "driver_number", "predicted_rank"]] \
                .sort_values(["race_id", "predicted_rank"])

        logging.info("Predicted Finishing Order:")
        results.to_csv(output_path, index=False, mode='w')
        logging.info(results)
        return True
    return False
        