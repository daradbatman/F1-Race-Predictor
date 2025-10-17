
from fastapi.concurrency import asynccontextmanager
from src.data.build_dataset import build_historical_dataset
from src.services.scheduler import start_dynamic_scheduler
from src.services.endpoints import router
from src.models.train import train_model
import os

import threading
import uvicorn
from fastapi import FastAPI

def _background_startup():
    try:
        build_historical_dataset()
        train_model()
    except Exception as e:
        # surface errors to container logs
        print("startup error:", e)
    # start scheduler as a daemon so it doesn't block shutdown
    threading.Thread(target=start_dynamic_scheduler, daemon=True).start()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # run blocking startup work in a daemon thread so the server binds immediately
    t = threading.Thread(target=_background_startup, daemon=True)
    t.start()
    yield
    # optional cleanup can go here

app = FastAPI(lifespan=lifespan)
app.include_router(router)

if __name__ == "__main__":
    # local/dev convenience: run uvicorn (imports app above so lifespan fires)
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("src.__main__:app", host="0.0.0.0", port=port)