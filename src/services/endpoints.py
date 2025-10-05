from fastapi import APIRouter, HTTPException
import pandas as pd
import os

router = APIRouter()

PREDICTIONS_LOG = "data/predictions/prediction_log.csv"

@router.get("/predictions")
def get_predictions():
    if not os.path.exists(PREDICTIONS_LOG):
        raise HTTPException(status_code=404, detail="Predictions log not found")

    try:
        df = pd.read_csv(PREDICTIONS_LOG)
        return df.to_dict(orient="records")  # JSON list of dicts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading predictions: {str(e)}")