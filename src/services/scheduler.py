from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from src.models.train import train_model
from src.models.predictor import run_prediction
from src.data.build_dataset import build_historical_dataset
from src.models.evaluate import evaluate_model
import logging
import time
from datetime import timedelta

logging.basicConfig(
    filename="logs/scheduler.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def safe_job(job_fn):
    def wrapper():
        try:
            job_fn()
        except Exception as e:
            logging.exception(f"Job {job_fn.__name__} failed")
    return wrapper

def retrain_pipeline_job():
    """Rebuild dataset + retrain model as one atomic pipeline."""
    logging.info(f"\n[{datetime.now()}] Starting retrain pipeline...")

    # Step 1: Rebuild historical dataset
    logging.info("Rebuilding dataset...")
    build_historical_dataset()
    logging.info("Dataset rebuilt.")

    # Step 2: Retrain model
    logging.info("Retraining model...")
    train_model()
    logging.info("Model retrained and saved.")

    logging.info(f"[{datetime.now()}] Retrain pipeline complete.\n")

def predict_job():
    logging.info(f"\n[{datetime.now()}] Running prediction...")
    run_prediction()
    logging.info("Predictions logged.")

def evaluate_job():
    logging.info(f"\n[{datetime.now()}] Evaluating model...")
    eval_report = evaluate_model()
    logging.info("Evaluation Report:", eval_report)

def start_dynamic_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.start()

    schedule_jobs(scheduler)

    scheduler.start()
    logging.info("Scheduler started.")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        scheduler.shutdown()
        logging.info("Scheduler stopped.")



def schedule_jobs(scheduler):
    scheduler.add_job(
        safe_job(predict_job),
        "cron",
        day_of_week="sat",
        hour=20,
        minute=30
    )
    scheduler.add_job(
        safe_job(retrain_pipeline_job),
        "cron",
        day_of_week="mon",
        hour=0,
        minute=0
    )
    scheduler.add_job(
        safe_job(evaluate_job),
        "cron",
        day_of_week="mon",
        hour=0,
        minute=0
    )