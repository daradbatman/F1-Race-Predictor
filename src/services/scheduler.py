from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from src.data import train_and_save_model, run_prediction
from src.data.build_dataset import build_historical_dataset
from src.models.evaluate import evaluate_model

PREDICTIONS_LOG = "data/predictions/predictions_log.csv"

def retrain_pipeline_job():
    """Rebuild dataset + retrain model as one atomic pipeline."""
    print(f"\n[{datetime.now()}] Starting retrain pipeline...")

    # Step 1: Rebuild historical dataset
    print("Rebuilding dataset...")
    build_historical_dataset()
    print("Dataset rebuilt.")

    # Step 2: Retrain model
    print("Retraining model...")
    train_and_save_model()
    print("Model retrained and saved.")

    print(f"[{datetime.now()}] Retrain pipeline complete.\n")

def predict_job():
    print(f"\n[{datetime.now()}] Running prediction...")
    results = run_prediction()

    # Save predictions
    results["timestamp"] = datetime.now()
    results.to_csv(PREDICTIONS_LOG, mode="a", index=False, header=False)
    print(f"Predictions logged to {PREDICTIONS_LOG}")

def evaluate_job():
    print(f"\n[{datetime.now()}] Evaluating model...")
    df_eval, eval_report = evaluate_model()
    print("Evaluation Report:", eval_report)

def start_scheduler():
    scheduler = BackgroundScheduler()

    # Schedule jobs (example times — adjust for actual F1 weekend timeline)
    scheduler.add_job(predict_job, "cron", day_of_week="sat", hour=16)         # Saturday 4PM after quali
    scheduler.add_job(retrain_pipeline_job, "cron", day_of_week="sun", hour=20) # Sunday 8PM after race
    scheduler.add_job(evaluate_job, "cron", day_of_week="sun", hour=23)         # Sunday 11PM after race

    scheduler.start()
    print("Scheduler started. Jobs will run automatically.")

    # Keep the app running
    try:
        while True:
            pass
    except KeyboardInterrupt:
        scheduler.shutdown()
        print("Scheduler stopped.")
