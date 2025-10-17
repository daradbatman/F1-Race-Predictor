from fastapi import APIRouter, HTTPException
import pandas as pd
import os
from src.models.predictor import run_prediction

router = APIRouter()

PREDICTIONS_LOG = "data/predictions/prediction_log.csv"

@router.get("/predictions")
def get_predictions():
    if not os.path.exists(PREDICTIONS_LOG):
        raise HTTPException(status_code=404, detail="Predictions has not been made yet.")

    try:
        is_successful = run_prediction()
        if is_successful:
            df = pd.read_csv(PREDICTIONS_LOG)
            return df.to_dict(orient="records")  # JSON list of dicts
        else:
            raise HTTPException(status_code=503, detail="Qualifying session data is not available yet to generate prediction.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading predictions: {str(e)}")