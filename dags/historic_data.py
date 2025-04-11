import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import logging
import json
import os
from tqdm import tqdm
from typing import Dict, List, Optional, Set, Union
from sklearn.preprocessing import MinMaxScaler
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('weather_data_fetch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OpenMeteoHistoricFetcher:
    def __init__(self):
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        self.daily_params = ["precipitation_sum"]
        self.hourly_params = [
            "temperature_2m", "precipitation", "wind_gusts_10m", 
            "pressure_msl", "dew_point_2m", "soil_moisture_0_to_7cm",
            "relative_humidity_2m", "wind_speed_10m", "cloud_cover",
            "snow_depth", "soil_temperature_0_to_7cm","precipitation_probability","evapotranspiration","snowfall"
        ]
        self.settings = {"timezone": "auto"}
        self.state_file = "weather_fetcher_state.json"
        self.processed_scenes: Set[str] = set()
        self.completed_batches: int = 0
        self.load_state()

    def load_state(self) -> None:
        """Load saved state including processed scenes"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.processed_scenes = set(state.get('processed_scenes', []))
                    self.completed_batches = state.get('completed_batches', 0)
                logger.info(f"Loaded state: {len(self.processed_scenes)} processed scenes, {self.completed_batches} batches")
        except Exception as e:
            logger.warning(f"Failed to load state: {str(e)}")
            self.processed_scenes = set()
            self.completed_batches = 0

    def save_state(self) -> None:
        """Save current state including processed scenes"""
        try:
            state = {
                'processed_scenes': list(self.processed_scenes),
                'completed_batches': self.completed_batches,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {str(e)}")

    def fetch_data(self, lat: float, lon: float, start_date: datetime, end_date: datetime) -> Optional[dict]:
        """Fetch weather data from archive API with rate limiting"""
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "daily": ",".join(self.daily_params),
            "hourly": ",".join(self.hourly_params),
            **self.settings
        }
        
        try:
            logger.debug(f"Fetching data for {lat},{lon} from {start_date} to {end_date}")
            response = requests.get(self.base_url, params=params, timeout=30)
            
            if response.status_code == 429:
                retry_after = response.headers.get('Retry-After', 'unknown')
                logger.warning(f"Rate limited. Retry after: {retry_after} seconds")
                return None
                
            response.raise_for_status()
            data = response.json()
            
            if not all(key in data for key in ["hourly", "daily"]):
                logger.warning("API response missing required sections")
                return None
                
            return data
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"API request failed: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to decode API response: {str(e)}")
            return None

def safe_extract(data: Union[dict, list], path: List[Union[str, int]], default=None):
    """Safely extract nested data from dictionary or list"""
    try:
        for key in path:
            if isinstance(key, int):
                if isinstance(data, (list, tuple)) and -len(data) <= key < len(data):
                    data = data[key]
                else:
                    return default
            else:
                data = data.get(key, {}) if isinstance(data, dict) else default
            if data is None:
                return default
        return data
    except (KeyError, TypeError, IndexError, AttributeError):
        return default

def process_openmeteo_data(json_data: dict, scene_id: str, acquisition_date: str) -> Optional[dict]:
    """Process API response into features with robust error handling"""
    if not json_data:
        return None
    
    try:
        # Validate response structure
        if not all(k in json_data for k in ["hourly", "daily"]):
            logger.warning(f"Missing sections in response for {scene_id}")
            return None
            
        hourly_data = json_data["hourly"]
        daily_data = json_data["daily"]
        
        # Find index for target time (00:00)
        target_time = f"{acquisition_date}T00:00"
        hourly_times = hourly_data.get("time", [])
        
        try:
            time_index = hourly_times.index(target_time)
        except ValueError:
            # Find closest hour if exact time not available
            try:
                time_obj = datetime.strptime(target_time, "%Y-%m-%dT%H:%M")
                min_diff = float('inf')
                best_index = None
                
                for i, t in enumerate(hourly_times):
                    if not t.startswith(acquisition_date):
                        continue
                    try:
                        t_obj = datetime.strptime(t, "%Y-%m-%dT%H:%M")
                        diff = abs((t_obj - time_obj).total_seconds())
                        if diff < min_diff:
                            min_diff = diff
                            best_index = i
                    except ValueError:
                        continue
                
                if best_index is not None:
                    time_index = best_index
                    target_time = hourly_times[time_index]
                else:
                    logger.warning(f"No valid time found for {scene_id} on {acquisition_date}")
                    return None
            except Exception as e:
                logger.warning(f"Time processing error for {scene_id}: {str(e)}")
                return None
                
        # Get daily index
        daily_dates = daily_data.get("time", [])
        try:
            daily_index = daily_dates.index(acquisition_date)
        except ValueError:
            daily_index = 0
            if daily_dates:  # Only warn if there actually are daily dates
                logger.debug(f"Date {acquisition_date} not in daily data for {scene_id}")

        features = {
            "scene_id": scene_id,
            "acquisition_date": acquisition_date,
            
            # Daily data
            "precip_sum_1d": safe_extract(daily_data, ["precipitation_sum", daily_index]),
            "precip_sum_2d": safe_extract(daily_data, ["precipitation_sum", daily_index + 1]) if daily_index + 1 < len(daily_dates) else None,
            
            # Hourly data
            "temp_2m": safe_extract(hourly_data, ["temperature_2m", time_index]),
            "precipitation_probability": safe_extract(hourly_data,['precipitation_probability', time_index]),
            "evapotranspiration": safe_extract(hourly_data, ["evapotranspiration", time_index]),
            "snowfall": safe_extract(hourly_data, ["snowfall", time_index]),
            "precip_hourly": safe_extract(hourly_data, ["precipitation", time_index]),
            "wind_gusts_10m": safe_extract(hourly_data, ["wind_gusts_10m", time_index]),
            "pressure_msl": safe_extract(hourly_data, ["pressure_msl", time_index]),
            "dewpoint_2m": safe_extract(hourly_data, ["dew_point_2m", time_index]),
            "soil_moisture": safe_extract(hourly_data, ["soil_moisture_0_to_7cm", time_index]),
            "humidity_2m": safe_extract(hourly_data, ["relative_humidity_2m", time_index]),
            "wind_speed_10m": safe_extract(hourly_data, ["wind_speed_10m", time_index]),
            "cloud_cover": safe_extract(hourly_data, ["cloud_cover", time_index]),
            "snow_depth": safe_extract(hourly_data, ["snow_depth", time_index]),
            "soil_temp": safe_extract(hourly_data, ["soil_temperature_0_to_7cm", time_index]),
            
            # Metadata
            "data_timestamp": target_time,
            "batch_number": fetcher.completed_batches + 1  # Current batch being processed
        }
        
        return features
        
    except Exception as e:
        logger.error(f"Error processing data for {scene_id} on {acquisition_date}: {str(e)}", exc_info=True)
        return None

def clean_weather_data():
    """Clean and prepare weather data"""
    weather_df = pd.read_csv("/opt/airflow/data/store/combined_historic_weather_features.csv")
    # Supprimer les lignes avec des valeurs manquantes
    weather_df = weather_df.dropna()
    
    # Supprimer les doublons basés sur scene_id et acquisition_date
    weather_df = weather_df.drop_duplicates(subset=["scene_id", "acquisition_date"])
    
    # Optionnel: Traiter les valeurs aberrantes
    # Exemple: Enlever les températures inférieures à -50°C ou supérieures à 50°C
    weather_df = weather_df[(weather_df["temp_2m"] >= -50) & (weather_df["temp_2m"] <= 50)]
    
    # Exemple de suppression des valeurs de précipitation improbables
    weather_df = weather_df[(weather_df["precip_hourly"] >= 0) & (weather_df["precip_hourly"] <= 500)]
    
    return weather_df

def normalize_weather_data():
    """Normalize the weather data to a standard scale"""
    weather_df = pd.read_csv("/opt/airflow/store/historic_data.csv")
    # Initialiser un normaliseur (par exemple, MinMaxScaler)
    scaler = MinMaxScaler()
    
    # Colonnes à normaliser
    columns_to_normalize = [
        "temp_2m", "precip_hourly", "wind_gusts_10m", "pressure_msl",
        "dewpoint_2m", "soil_moisture", "humidity_2m", "wind_speed_10m", 
        "cloud_cover", "snow_depth", "soil_temp"
    ]
    
    # Appliquer la normalisation
    weather_df[columns_to_normalize] = scaler.fit_transform(weather_df[columns_to_normalize])
    
    return weather_df

def fetch_flood_weather_data(scene_metadata: Dict, batch_size: int = 200) -> pd.DataFrame:
    """Process data in batches with enhanced state management"""
    global fetcher
    fetcher = OpenMeteoHistoricFetcher()
    weather_data = []
    
    # Filter scenes to only unprocessed ones
    scenes_to_process = [
        (sid, meta) for sid, meta in scene_metadata.items() 
        if sid not in fetcher.processed_scenes
    ]
    
    total_scenes = len(scenes_to_process)
    logger.info(f"Found {total_scenes} scenes to process (already processed {len(fetcher.processed_scenes)})")
    
    if not scenes_to_process:
        logger.warning("No scenes left to process")
        return pd.DataFrame()
    
    # Process one batch
    scenes_in_batch = min(batch_size, len(scenes_to_process))
    logger.info(f"Processing batch {fetcher.completed_batches + 1} with {scenes_in_batch} scenes")
    
    with tqdm(total=scenes_in_batch, desc=f"Batch {fetcher.completed_batches + 1}") as pbar:
        for scene_id, metadata in scenes_to_process[:scenes_in_batch]:
            try:
                # Extract coordinates with validation
                coords = metadata.get("geo", {}).get("coordinates", [[[0, 0]]])
                if not coords or not coords[0]:
                    logger.warning(f"Invalid coordinates for scene {scene_id}")
                    continue
                    
                try:
                    coords_array = np.array(coords[0])
                    if coords_array.size < 2:
                        logger.warning(f"Insufficient coordinates for scene {scene_id}")
                        continue
                        
                    lon, lat = coords_array.mean(axis=0)[:2]
                    logger.debug(f"Processing scene {scene_id} at ({lat:.4f}, {lon:.4f})")
                except Exception as e:
                    logger.warning(f"Coordinate processing error for {scene_id}: {str(e)}")
                    continue
                
                # Process each acquisition date
                for acq_key, acquisition in metadata.items():
                    if not isinstance(acq_key, str) or not acq_key.isdigit():
                        continue
                        
                    date_str = acquisition.get("date")
                    if not date_str:
                        continue
                        
                    try:
                        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                        start_date = date_obj - timedelta(days=1)
                        
                        api_data = fetcher.fetch_data(lat, lon, start_date, date_obj)
                        
                        if api_data:
                            features = process_openmeteo_data(api_data, scene_id, date_str)
                            if features:
                                features["flood_label"] = acquisition.get("FLOODING", False)
                                weather_data.append(features)
                    except ValueError as e:
                        logger.warning(f"Invalid date format for scene {scene_id}: {date_str}")
                    except Exception as e:
                        logger.warning(f"Unexpected error processing {scene_id}: {str(e)}")
                
                # Mark scene as processed
                fetcher.processed_scenes.add(scene_id)
                
            except Exception as e:
                logger.error(f"Critical error processing scene {scene_id}: {str(e)}", exc_info=True)
            finally:
                pbar.update(1)
    
    # Finalize batch processing
    if scenes_in_batch > 0:
        fetcher.completed_batches += 1
        fetcher.save_state()
        logger.info(f"Completed batch {fetcher.completed_batches}")
    
    # Save results
    weather_df = pd.DataFrame(weather_data) if weather_data else pd.DataFrame()
    
    if not weather_df.empty:
        output_file = f"weather_features_batch_{fetcher.completed_batches}.csv"
        weather_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(weather_df)} records to {output_file}")
    else:
        logger.warning("No weather data was processed in this batch")
    
    return weather_df

def combine_results(output_file: str = "/opt/airflow/data/store/combined_historic_weather_features.csv") -> None:
    """Combine all batch files into one with validation"""
    import glob
    all_files = sorted(glob.glob("weather_features_batch_*.csv"))
    
    if not all_files:
        logger.warning("No batch files found to combine")
        return
    
    logger.info(f"Combining {len(all_files)} batch files...")
    
    try:
        # Read and validate each file
        dfs = []
        for file in all_files:
            try:
                df = pd.read_csv(file)
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to read {file}: {str(e)}")
        
        if not dfs:
            logger.warning("No valid data to combine")
            return
            
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Validate combined data
        if combined_df.empty:
            logger.warning("Combined dataframe is empty")
            return
            
        # Check for duplicate scene_ids
        dupes = combined_df.duplicated(subset=["scene_id", "acquisition_date"], keep=False)
        if dupes.any():
            logger.warning(f"Found {dupes.sum()} duplicate scene entries")
            combined_df = combined_df.drop_duplicates(subset=["scene_id", "acquisition_date"], keep="first")
        
        combined_df.to_csv(output_file, index=False)
        logger.info(f"Successfully combined data into {output_file} ({len(combined_df)} records)")
        
    except Exception as e:
        logger.error(f"Failed to combine results: {str(e)}", exc_info=True)

