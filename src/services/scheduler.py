from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from src.models.train import train_model
from src.data.build_dataset import build_historical_dataset
from src.models.evaluate import evaluate_model
import logging
import time
import os

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_LEVEL_VALUE = getattr(logging, _LOG_LEVEL, logging.INFO)
if not logging.getLogger().handlers:
    logging.basicConfig(level=_LOG_LEVEL_VALUE, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def safe_job(job_fn):
    def wrapper():
        try:
            job_fn()
        except Exception as e:
            logger.exception(f"Job {job_fn.__name__} failed")
    return wrapper

def retrain_pipeline_job():
    """Rebuild dataset + retrain model as one atomic pipeline."""
    logger.info(f"\n[{datetime.now()}] Starting retrain pipeline...")

    # Step 1: Rebuild historical dataset
    logger.info("Rebuilding dataset...")
    build_historical_dataset()
    logger.info("Dataset rebuilt.")

    # Step 2: Retrain model
    logger.info("Retraining model...")
    train_model()
    logger.info("Model retrained and saved.")

    logger.info(f"[{datetime.now()}] Retrain pipeline complete.\n")

def evaluate_job():
    logger.info(f"\n[{datetime.now()}] Evaluating model...")
    eval_report = evaluate_model()
    logger.info("Evaluation Report:", eval_report)

def start_dynamic_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.start()

    schedule_jobs(scheduler)

    scheduler.start()
    logger.info("Scheduler started.")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        scheduler.shutdown()
        logger.info("Scheduler stopped.")



def schedule_jobs(scheduler):
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