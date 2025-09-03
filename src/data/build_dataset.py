import pandas as pd
from src.data.open_F1_service import fetch_meetings, fetch_sessions, fetch_results, fetch_driver, fetch_starting_positions, fetch_laps, fetch_driver_stints, fetch_weather, fetch_latest_meeting


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

def summarize_stints(stints: pd.DataFrame) -> dict:
    """Aggregate stint data into race-level features for one driver."""
    if stints.empty:
        return {
            "num_stints": None,
            "num_pit_stops": None,
            "avg_stint_length": None,
            "max_stint_length": None,
            "used_soft": 0,
            "used_medium": 0,
            "used_hard": 0,
            "avg_tyre_age_start": None,
        }

    # Compute stint lengths
    stint_lengths = (stints["lap_end"] - stints["lap_start"] + 1).tolist()

    # Compound usage
    compounds = stints["compound"].str.upper().unique().tolist()

    return {
        "num_stints": len(stints),
        "num_pit_stops": len(stints) - 1,
        "avg_stint_length": sum(stint_lengths) / len(stint_lengths),
        "max_stint_length": max(stint_lengths),
        "used_soft": int("SOFT" in compounds),
        "used_medium": int("MEDIUM" in compounds),
        "used_hard": int("HARD" in compounds),
        "avg_tyre_age_start": stints["tyre_age_at_start"].mean(),
    }

def summarize_laps(laps: pd.DataFrame) -> dict:
    """Aggregate lap data into driver-level race features."""
    if laps.empty:
        return {
            "total_laps": 0,
            "avg_lap_time": None,
            "best_lap_time": None,
            "std_lap_time": None,
            "laps_with_pit": 0,
            "avg_sector1": None,
            "avg_sector2": None,
            "avg_sector3": None,
            "avg_speed_trap": None,
            "avg_i1_speed": None,
            "avg_i2_speed": None,
        }

    return {
        "total_laps": laps["lap_number"].max(),
        "avg_lap_time": laps["lap_duration"].mean(),
        "best_lap_time": laps["lap_duration"].min(),
        "std_lap_time": laps["lap_duration"].std(),  # consistency measure
        "laps_with_pit": laps["is_pit_out_lap"].sum(),

        # Sector averages
        "avg_sector1": laps["duration_sector_1"].mean(),
        "avg_sector2": laps["duration_sector_2"].mean(),
        "avg_sector3": laps["duration_sector_3"].mean(),

        # Speeds
        "avg_speed_trap": laps["st_speed"].mean(),
        "avg_i1_speed": laps["i1_speed"].mean(),
        "avg_i2_speed": laps["i2_speed"].mean(),
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
            weather = fetch_weather(meeting["meeting_key"], race_session["session_key"])

            weather_features = summarize_weather(weather)

            for _, res in results.iterrows():
                driver = fetch_driver(res["driver_number"], race_session["session_key"])
                if driver.empty:
                    continue

                starting_pos_row = starting_grid[starting_grid["driver_number"] == res["driver_number"]]
                starting_position = (
                    starting_pos_row["position"].iloc[0] if not starting_pos_row.empty else 0
                )

                # --- Handle finishing position & DNF flag ---
                pos_raw = res.get("position")
                if pd.isna(pos_raw) or str(pos_raw).upper() in ["DNF", "DNS", "DSQ"]:
                    finishing_position = 31  # worse than P20
                    dnf_flag = 1
                else:
                    try:
                        finishing_position = int(pos_raw)
                        dnf_flag = 0
                    except (ValueError, TypeError):
                        finishing_position = 31
                        dnf_flag = 1

                row = {
                    "season": meeting["year"],
                    "race": meeting["meeting_name"],
                    "circuit": meeting["location"],
                    "date": race_session["date_start"],

                    "driver_number": res["driver_number"],
                    "driver_name": driver["full_name"].iloc[0],
                    "constructor": driver["team_name"].iloc[0],

                    "starting_position": starting_position,
                    "finishing_position": finishing_position,
                    "dnf": dnf_flag,

                    **weather_features,
                }
                all_races.append(row)

        except Exception as e:
            print(f"Failed meeting {meeting['meeting_name']}: {e}")
    
    int_cols = ["season", "driver_number", "starting_position", "finishing_position", "dnf", "relevance_label"]
    df = pd.DataFrame(all_races)
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)


    df.to_csv("data/processed/features.csv", index=False)
    print(f"Saved dataset with {len(df)} rows")

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
            print("No qualifying session found for latest meeting.")
            return
        qualifying_session = qualifying_sessions.iloc[0]
        starting_grid = fetch_starting_positions(qualifying_session["session_key"])
        weather = fetch_weather(qualifying_session["meeting_key"], qualifying_session["session_key"]) 
        weather_features = summarize_weather(weather)

        for _, driver in starting_grid.iterrows():
            driver_info = fetch_driver(driver["driver_number"], qualifying_session["session_key"])
            #laps = fetch_laps(qualifying_session["meeting_key"], qualifying_session["session_key"], driver["driver_number"])
            #laps_features = summarize_laps(laps)
            if driver_info.empty:
                continue

            starting_pos_row = starting_grid[starting_grid["driver_number"] == driver["driver_number"]]
            starting_position = starting_pos_row["position"].iloc[0] if not starting_pos_row.empty else 0

            row = {
                "season": meeting["year"],
                "race": meeting["meeting_name"],
                "circuit": meeting["location"],
                "date": qualifying_session["date_start"],

                "driver_number": driver["driver_number"],
                "driver_name": driver_info["full_name"].iloc[0],
                "constructor": driver_info["team_name"].iloc[0],

                "starting_position": starting_position,
                "dnf": 0,

                **weather_features,
            }
            races.append(row)

    except Exception as e:
        print(f"Failed to build latest race dataset: {e}")


    df = pd.DataFrame(races)
    int_cols = ["season", "driver_number", "starting_position", "finishing_position", "dnf"]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df.to_csv("data/processed/latest.csv", index=False)
    print(f"Saved dataset with {len(df)} rows")

    return df
    
