import requests
import pandas as pd
import os
import time
import threading
import collections
import logging

BASE_URL = "https://api.openf1.org/v1"

# logging
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_LEVEL_VALUE = getattr(logging, _LOG_LEVEL, logging.INFO)
if not logging.getLogger().handlers:
    logging.basicConfig(level=_LOG_LEVEL_VALUE, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls = collections.deque()
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            # pop expired timestamps
            while self.calls and self.calls[0] <= now - self.period:
                self.calls.popleft()
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                logger.debug("RateLimiter: allowed call (calls=%d)", len(self.calls))
                return
            # need to wait until oldest expires
            wait_for = (self.calls[0] + self.period) - now
            logger.debug("RateLimiter: reached max_calls=%d; waiting %.3fs", self.max_calls, wait_for)
        time.sleep(wait_for)
        # after sleeping, register the call
        self.wait()

# configurable via environment variables
_MAX_CALLS = int(os.getenv("RATE_LIMIT_MAX_CALLS", "10"))  # default 10 calls
_PERIOD = float(os.getenv("RATE_LIMIT_PERIOD", "1"))       # default per 1 second
_MAX_RETRIES = int(os.getenv("RATE_LIMIT_MAX_RETRIES", "3"))
_BACKOFF_FACTOR = float(os.getenv("RATE_LIMIT_BACKOFF_FACTOR", "1.5"))

_rate_limiter = RateLimiter(_MAX_CALLS, _PERIOD)
_session = requests.Session()

def _safe_get(url, params=None, timeout=10):
    """Rate-limited GET with basic 429 retry/backoff (honors Retry-After)."""
    for attempt in range(_MAX_RETRIES + 1):
        _rate_limiter.wait()
        logger.debug("HTTP GET attempt=%d url=%s params=%s", attempt + 1, url, params)
        try:
            resp = _session.get(url, params=params, timeout=timeout)
        except Exception:
            logger.exception("Request failed (attempt=%d) url=%s params=%s", attempt + 1, url, params)
            # follow same retry/backoff behaviour
            resp = None
        if resp is None:
            sleep_for = (_BACKOFF_FACTOR ** attempt)
            logger.debug("Sleeping %.3fs after exception before retry", sleep_for)
            time.sleep(sleep_for)
            continue
        logger.debug("Response status=%d for url=%s", resp.status_code, url)
        if resp.status_code != 429:
            if resp.status_code >= 400:
                logger.warning("Non-429 HTTP status %d for %s", resp.status_code, url)
            return resp
        # handle 429: check Retry-After header
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                sleep_for = float(retry_after)
            except ValueError:
                # Retry-After could be an HTTP-date; fall back to exponential backoff
                sleep_for = (_BACKOFF_FACTOR ** attempt)
        else:
            sleep_for = (_BACKOFF_FACTOR ** attempt)
        logger.warning("Received 429 for %s; retry-after=%s; sleeping %.3fs (attempt=%d)", url, retry_after, sleep_for, attempt + 1)
        time.sleep(sleep_for)
    # last attempt result (could still be 429)
    return resp

def fetch_meetings():
    logger.info("fetch_meetings")
    response = _safe_get(f"{BASE_URL}/meetings")
    response.raise_for_status()
    meetings = response.json()
    return pd.DataFrame(meetings)

def fetch_sessions(meeting_key: int):
    logger.info("fetch_sessions meeting_key=%s", meeting_key)
    response = _safe_get(f"{BASE_URL}/sessions", params={"meeting_key": meeting_key})
    response.raise_for_status()
    sessions = response.json()
    return pd.DataFrame(sessions)

def fetch_starting_positions(session_key: int):
    logger.info("fetch_starting_positions session_key=%s", session_key)
    response = _safe_get(f"{BASE_URL}/starting_grid", params={"session_key": session_key})
    response.raise_for_status()
    starting_positions = response.json()
    return pd.DataFrame(starting_positions)

def fetch_results(session_key: int):
    logger.info("fetch_results session_key=%s", session_key)
    response = _safe_get(f"{BASE_URL}/session_result", params={"session_key": session_key})
    response.raise_for_status()
    results = response.json()
    return pd.DataFrame(results)

def fetch_driver(driver_number: int, session_key: int):
    logger.info("fetch_driver driver_number=%s session_key=%s", driver_number, session_key)
    response = _safe_get(f"{BASE_URL}/drivers", params={"driver_number": driver_number, "session_key": session_key})
    response.raise_for_status()
    drivers = response.json()
    return pd.DataFrame(drivers)

def fetch_weather(meeting_key: int, session_key: int):
    logger.info("fetch_weather meeting_key=%s session_key=%s", meeting_key, session_key)
    response = _safe_get(f"{BASE_URL}/weather", params={"meeting_key": meeting_key, "session_key": session_key})
    response.raise_for_status()
    weather = response.json()
    return pd.DataFrame(weather)

def fetch_latest_meeting():
    logger.info("fetch_latest_meeting")
    response = _safe_get(f"{BASE_URL}/meetings", params={"meeting_key": "latest"})
    response.raise_for_status()
    meeting = response.json()
    return pd.DataFrame(meeting)

def fetch_latest_session_results():
    logger.info("fetch_latest_session_results")
    response = _safe_get(f"{BASE_URL}/session_result", params={"session_key": "latest"})
    response.raise_for_status()
    sessions = response.json()
    return pd.DataFrame(sessions)