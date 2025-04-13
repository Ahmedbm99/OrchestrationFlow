import os
import json
import requests
from typing import List, Dict, Set
from datetime import datetime
import pandas as pd
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
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.processed_scenes = set(state.get('processed_scenes', []))
                logger.info(f"Loaded state: {len(self.processed_scenes)} processed scenes")
        except Exception as e:
            logger.warning(f"Failed to load state: {str(e)}")
            self.processed_scenes = set()

    def save_state(self) -> None:
        try:
            state = {
                'processed_scenes': list(self.processed_scenes),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {str(e)}")

    def fetch_weather(self, lat: float, lon: float, start: str, end: str) -> Dict:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
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


def process_openmeteo_data(data: Dict, scene_id: str) -> List[Dict]:
    if "hourly" not in data:
        logger.warning(f"No hourly data found for scene {scene_id}")
        return []

    result = []
    hourly = data["hourly"]
    times = hourly.get("time", [])

    for i, timestamp in enumerate(times):
        row = {
            "scene_id": scene_id,
            "timestamp": timestamp
        }
        for param in hourly:
            if param != "time":
                row[param] = hourly[param][i]
        result.append(row)
    return result

def fetch_flood_weather_data(metadata_file: str, output_file: str) -> None:
    fetcher = OpenMeteoHistoricFetcher()

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    weather_data = []

    # Iterate over each folder in the metadata
    for folder_id, folder_data in tqdm(metadata.items(), desc="Fetching weather data"):
        # Check for scenes in the folder
        for scene_key, scene in folder_data.items():
            if scene_key == "count" or scene_key == "folder" or scene_key == "geo" or scene_key == "urls":
                continue  # Skip non-scene fields

            scene_id = f"{folder_id}_{scene_key}"
            if fetcher.is_processed(scene_id):
                continue

            lat, lon = None, None
            # Extract geographical coordinates from the "geo" field if available
            if 'geo' in folder_data:
                lat, lon = folder_data['geo']['coordinates'][0][0][1], folder_data['geo']['coordinates'][0][0][0]
            
            date = scene.get("date")
            flooding = scene.get("FLOODING")
            orbit = scene.get("orbit")
            filename = scene.get("filename")
            start = date  # Using the same date as both start and end for simplicity, adjust if needed
            end = date

            if not all([lat, lon, start, end]):
                logger.warning(f"Incomplete metadata for scene {scene_id}, skipping.")
                continue

            # Fetch weather data from OpenMeteo
            data = fetcher.fetch_weather(lat, lon, start, end)
            processed = process_openmeteo_data(data, scene_id)

            if processed:
                weather_data.extend(processed)
                fetcher.mark_as_processed(scene_id)

    if weather_data:
        df = pd.DataFrame(weather_data)
        df.to_csv(output_file, index=False)
        logger.info(f"Saved weather data to {output_file}")
    else:
        logger.info("No new data to save.")

    fetcher.save_state()

def clean_weather_data():
        """Clean and prepare weather data"""
        weather_df = pd.read_csv("/opt/airflow/data/store/combined_historic_weather_features.csv")
    # Supprimer les lignes avec des valeurs manquantes
        weather_df = weather_df.dropna()
    
    # Supprimer les doublons basés sur scene_id et acquisition_date
        weather_df = weather_df.drop_duplicates(subset=["scene_id", "acquisition_date"])
    
    # Optionnel: Traiter les valeurs aberrantes
    # Exemple: Enlever les températures inférieures à -50°C ou supérieures à 50°C
        weather_df = weather_df[(weather_df["temperature_2m"] >= -50) & (weather_df["temperature_2m"] <= 50)]
    
    # Exemple de suppression des valeurs de précipitation improbables
        weather_df = weather_df[(weather_df["precipitation"] >= 0) & (weather_df["precipitation"] <= 500)]
    
        return weather_df