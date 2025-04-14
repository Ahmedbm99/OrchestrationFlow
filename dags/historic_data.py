import os
import sys
import json
import requests
import numpy as np
import pandas as pd
from typing import List, Dict, Set
from datetime import datetime, timedelta
from tqdm import tqdm
import logging

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenMeteoHistoricFetcher:
    def __init__(self):
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        self.daily_params = ["precipitation_sum"]
        self.hourly_params = [
            "temperature_2m", "precipitation", "wind_gusts_10m",
            "pressure_msl", "dew_point_2m", "soil_moisture_0_to_7cm",
            "relative_humidity_2m", "wind_speed_10m", "cloud_cover",
            "snow_depth", "soil_temperature_0_to_7cm",
            "precipitation_probability", "evapotranspiration", "snowfall"
        ]
        self.settings = {"timezone": "auto"}
        self.state_file = "weather_fetcher_state.json"
        self.processed_scenes: Set[str] = set()
        self.load_state()

    def load_state(self) -> None:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.processed_scenes = set(state.get('processed_scenes', []))
                logger.info(f"Loaded state: {len(self.processed_scenes)} processed scenes")
            except Exception as e:
                logger.warning(f"Failed to load state: {str(e)}")

    def save_state(self) -> None:
        try:
            with open(self.state_file, 'w') as f:
                json.dump({
                    'processed_scenes': list(self.processed_scenes),
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {str(e)}")

    def fetch_data(self, lat: float, lon: float, start: datetime, end: datetime) -> Dict:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "daily": ",".join(self.daily_params),
            "hourly": ",".join(self.hourly_params),
            **self.settings
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Request failed for {lat},{lon} ({start} to {end}): {e}")
            return {}

    def is_processed(self, scene_id: str) -> bool:
        return scene_id in self.processed_scenes

    def mark_as_processed(self, scene_id: str) -> None:
        self.processed_scenes.add(scene_id)


def process_openmeteo_data(data: Dict, scene_id: str, acquisition_date: str) -> List[Dict]:
    if "hourly" not in data:
        logger.warning(f"No hourly data found for scene {scene_id}")
        return []

    result = []
    hourly = data["hourly"]
    times = hourly.get("time", [])

    for i, timestamp in enumerate(times):
        row = {
            "scene_id": scene_id,
            "timestamp": timestamp,
            "acquisition_date": acquisition_date
        }
        for param in hourly:
            if param != "time":
                row[param] = hourly[param][i]
        result.append(row)
    return result


def save_partial_results(data_rows: List[Dict], prefix: str = "") -> None:
    if data_rows:
        df = pd.DataFrame(data_rows)
        output_path = f"/opt/airflow/data/store/{prefix}combined_historic_weather_features.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved weather data to: {output_path}")


def fetch_flood_weather_data(scene_metadata: Dict) -> pd.DataFrame:
    fetcher = OpenMeteoHistoricFetcher()
    all_weather_data = []

    scene_items = [(sid, meta) for sid, meta in scene_metadata.items()
                   if isinstance(sid, str) and sid.isdigit()]

    try:
        with tqdm(total=len(scene_items), desc="Processing Scenes") as scene_pbar:
            for scene_id, metadata in scene_items:
                if fetcher.is_processed(scene_id):
                    scene_pbar.update(1)
                    continue
                try:
                    coords = metadata.get("geo", {}).get("coordinates", [[[0, 0]]])[0]
                    lon, lat = np.array(coords).mean(axis=0)

                    acq_items = [(k, v) for k, v in metadata.items() if k.isdigit()]
                    for acq_key, acquisition in acq_items:
                        date_str = acquisition.get("date")
                        if not date_str:
                            continue
                        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                        start_date = date_obj - timedelta(days=1)
                        end_date = date_obj

                        api_data = fetcher.fetch_data(lat, lon, start_date, end_date)

                        weather_features = process_openmeteo_data(
                            api_data, scene_id, date_str
                        )

                        for row in weather_features:
                            row["flood_label"] = acquisition.get("FLOODING", False)

                        all_weather_data.extend(weather_features)

                    fetcher.mark_as_processed(scene_id)

                except Exception as e:
                    logger.error(f"Error processing scene {scene_id}: {str(e)}", exc_info=True)
                finally:
                    scene_pbar.update(1)

        fetcher.save_state()

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)

    # Sauvegarde finale (toujours dans combined_historic_weather_features.csv)
    save_partial_results(all_weather_data)

    return pd.DataFrame(all_weather_data)


def clean_weather_data():
    """Clean and prepare weather data"""
    file_path = "/opt/airflow/data/store/combined_historic_weather_features.csv"
    if not os.path.exists(file_path):
        logger.warning(f"{file_path} not found.")
        return pd.DataFrame()

    weather_df = pd.read_csv(file_path)

    # Drop missing and duplicate entries
    weather_df = weather_df.dropna()
    weather_df = weather_df.drop_duplicates(subset=["scene_id", "acquisition_date"])

    # Outlier filtering
    weather_df = weather_df[(weather_df["temperature_2m"] >= -50) & (weather_df["temperature_2m"] <= 50)]
    weather_df = weather_df[(weather_df["precipitation"] >= 0) & (weather_df["precipitation"] <= 500)]

    return weather_df
