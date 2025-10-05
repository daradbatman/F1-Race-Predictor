
from src.data.build_dataset import build_historical_dataset
from src.services.scheduler import start_dynamic_scheduler
from src.services.endpoints import router

import threading
import uvicorn
from fastapi import FastAPI

def run_scheduler():
    start_dynamic_scheduler()

def run_api():
    app = FastAPI()
    app.include_router(router)
    uvicorn.run(app, host="0.0.0.0", port=8000)

def main():
    build_historical_dataset()
    # Start scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    # Start FastAPI app (blocking call)
    run_api()

if __name__ == "__main__":
    main()