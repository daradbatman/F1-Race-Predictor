import requests
import pandas as pd

BASE_URL = "https://api.openf1.org/v1"

def fetch_meetings():
    response = requests.get(f"{BASE_URL}/meetings")
    response.raise_for_status()
    meetings = response.json()
    return pd.DataFrame(meetings)

def fetch_sessions(meeting_key: int):
    response = requests.get(f"{BASE_URL}/sessions?meeting_key={meeting_key}")
    response.raise_for_status()
    sessions = response.json()
    return pd.DataFrame(sessions)

def fetch_starting_positions(session_key: int):
    response = requests.get(f"{BASE_URL}/starting_grid?session_key={session_key}")
    response.raise_for_status()
    starting_positions = response.json()
    return pd.DataFrame(starting_positions)

def fetch_results(session_key: int):
    response = requests.get(f"{BASE_URL}/session_result?session_key={session_key}")
    response.raise_for_status()
    results = response.json()
    return pd.DataFrame(results)

def fetch_driver(driver_number: int, session_key: int):
    response = requests.get(f"{BASE_URL}/drivers?driver_number={driver_number}&session_key={session_key}")
    response.raise_for_status()
    drivers = response.json()
    return pd.DataFrame(drivers)

def fetch_weather(meeting_key: int, session_key: int):
    response = requests.get(f"{BASE_URL}/weather?meeting_key={meeting_key}&session_key={session_key}")
    response.raise_for_status()
    weather = response.json()
    return pd.DataFrame(weather)

def fetch_latest_meeting():
    response = requests.get(f"{BASE_URL}/meetings?meeting_key=latest")
    response.raise_for_status()
    meeting = response.json()
    return pd.DataFrame(meeting)

def fetch_latest_session_results():
    response = requests.get(f"{BASE_URL}/session_result?session_key=latest")
    response.raise_for_status()
    sessions = response.json()
    return pd.DataFrame(sessions)
