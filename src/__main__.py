
from src.models.train import train_model
from src.data.build_dataset import build_historical_dataset, build_latest_race_dataset
from src.models.train import train_model
from src.models.predictor import run_prediction
from src.models.evaluate import evaluate_model

def main():
    #build_historical_dataset()
    #build_latest_race_dataset()
    train_model()
    run_prediction()
    evaluate_model()

if __name__ == "__main__":
    main()