import pandas as pd
from src.data.open_F1_service import fetch_meetings, fetch_sessions, fetch_results, fetch_driver, fetch_starting_positions, fetch_weather, fetch_latest_meeting
import logging
import os

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_LEVEL_VALUE = getattr(logging, _LOG_LEVEL, logging.INFO)
if not logging.getLogger().handlers:
    logging.basicConfig(level=_LOG_LEVEL_VALUE, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def summarize_weather(weather: pd.DataFrame) -> dict:
    """Aggregate weather samples into session-level features."""
    if weather.empty:
        return {
            "avg_track_temp": None,
            "max_track_temp": None,
            "min_track_temp": None,
            "avg_air_temp": None,
            "avg_humidity": None,
            "avg_pressure": None,
            "rain_occurrence": 0,  # 1 if it rained at least once
            "avg_wind_speed": None,
            "dominant_wind_dir": None,  # most frequent direction
        }

    return {
        "avg_track_temp": weather["track_temperature"].mean(),
        "max_track_temp": weather["track_temperature"].max(),
        "min_track_temp": weather["track_temperature"].min(),
        "avg_air_temp": weather["air_temperature"].mean(),
        "avg_humidity": weather["humidity"].mean(),
        "avg_pressure": weather["pressure"].mean(),
        "rain_occurrence": int(weather["rainfall"].sum() > 0),
        "avg_wind_speed": weather["wind_speed"].mean(),
        "dominant_wind_dir": weather["wind_direction"].mode().iloc[0]
        if not weather["wind_direction"].mode().empty
        else None,
    }

def build_historical_dataset(limit_year: int = 2023):
    """Build dataset with race-level features + results since limit_year"""
    meetings = fetch_meetings()
    meetings = meetings[meetings["year"] >= limit_year]

    all_races = []

    for _, meeting in meetings.iterrows():
        try:
            sessions = fetch_sessions(meeting["meeting_key"])
            qualifying_sessions = sessions[
                (sessions["session_type"] == "Qualifying") &
                (sessions["session_name"] == "Qualifying")
            ]
            race_sessions = sessions[
                (sessions["session_type"] == "Race") &
                (sessions["session_name"] == "Race")
            ]

            if race_sessions.empty or qualifying_sessions.empty:
                continue

            race_session = race_sessions.iloc[0]
            qualifying_session = qualifying_sessions.iloc[0]
            starting_grid = fetch_starting_positions(qualifying_session["session_key"])
            results = fetch_results(race_session["session_key"])
            weather = fetch_weather(meeting["meeting_key"], qualifying_session["session_key"])

            weather_features = summarize_weather(weather)

            for _, row in starting_grid.iterrows():
                driver = fetch_driver(row["driver_number"], qualifying_session["session_key"])
                if driver.empty:
                    logger.info(f"Driver {row['driver_number']} not found, skipping.")
                    continue

                starting_pos_row = starting_grid[starting_grid["driver_number"] == row["driver_number"]]
                starting_position = (
                    starting_pos_row["position"].iloc[0] if not starting_pos_row.empty else 21
                )

                # --- Handle finishing position & DNF flag ---
                result_row = results[results["driver_number"] == row["driver_number"]]
                if result_row.empty:
                    logger.info(f"Result for driver {row['driver_number']} not found, assuming DNF.")
                    continue

                pos_raw = result_row.get("position").iloc[0]
                try:
                    finishing_position = int(pos_raw)
                except (ValueError, TypeError):
                    finishing_position = 21

                row = {
                    "race_id": f"{meeting['year']}_{meeting['meeting_key']}",
                    "season": meeting["year"],
                    "race": meeting["meeting_name"],
                    "circuit": meeting["location"],
                    "date": race_session["date_start"],

                    "driver_number": row["driver_number"],
                    "driver_name": driver["full_name"].iloc[0],
                    "constructor": driver["team_name"].iloc[0],

                    "starting_position": starting_position,
                    "finishing_position": finishing_position,

                    **weather_features,
                }
                all_races.append(row)

        except Exception as e:
            logger.exception(f"Failed meeting {meeting['meeting_name']}: {e}")
    
    int_cols = ["season", "driver_number", "starting_position", "finishing_position", "dnf", "relevance_label"]
    df = pd.DataFrame(all_races)
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)


    df.to_csv("data/processed/features.csv", index=False)
    logger.info(f"Saved dataset with {len(df)} rows")

    return df

def build_latest_race_dataset():
    """Build dataset for the latest race only."""
    races = []
    try:
        latest_meeting = fetch_latest_meeting()
        meeting = latest_meeting.iloc[0]

        sessions = fetch_sessions(meeting["meeting_key"])
        qualifying_sessions = sessions[(sessions["session_type"] == "Qualifying") & (sessions["session_name"] == "Qualifying")]
        if qualifying_sessions.empty:
            logger.info("No qualifying session found for latest meeting.")
            return
        qualifying_session = qualifying_sessions.iloc[0]
        starting_grid = fetch_starting_positions(qualifying_session["session_key"])
        weather = fetch_weather(qualifying_session["meeting_key"], qualifying_session["session_key"]) 
        weather_features = summarize_weather(weather)

        for _, driver in starting_grid.iterrows():
            driver_info = fetch_driver(driver["driver_number"], qualifying_session["session_key"])

            if driver_info.empty:
                logger.info(f"Driver {driver['driver_number']} not found, skipping.")
                continue

            starting_pos_row = starting_grid[starting_grid["driver_number"] == driver["driver_number"]]
            starting_position = starting_pos_row["position"].iloc[0] if not starting_pos_row.empty else 0

            row = {
                "race_id": f"{meeting['year']}_{meeting['meeting_key']}",
                "season": meeting["year"],
                "race": meeting["meeting_name"],
                "circuit": meeting["location"],
                "date": qualifying_session["date_start"],

                "driver_number": driver["driver_number"],
                "driver_name": driver_info["full_name"].iloc[0],
                "constructor": driver_info["team_name"].iloc[0],

                "starting_position": starting_position,

                **weather_features,
            }
            races.append(row)
        df = pd.DataFrame(races)
        int_cols = ["season", "driver_number", "starting_position", "finishing_position", "dnf"]
        for c in int_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        df.to_csv("data/processed/latest.csv", index=False)
        logger.info(f"Saved dataset with {len(df)} rows")

        return df

    except Exception as e:
        logger.exception(f"Failed to build latest race dataset: {e}")

    
