import requests
import pandas as pd
import logging
from typing import Optional, Union, List
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('real_time_weather_fetch.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class OpenMeteoRealTimeFetcher:
    def __init__(self):
        self.latitude = 36.9598893
        self.longitude = 10.0095806
        self.api_url = "https://api.open-meteo.com/v1/forecast"
        self.settings = {"timezone": "GMT"}
        self.output_file = '/opt/airflow/data/store/weather_realtime_data.csv'

    def fetch_data(self) -> Optional[dict]:
        """
        Récupère les données météorologiques en temps réel à partir de l'API Open-Meteo.
        """
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "daily": "precipitation_sum",
            "hourly": "temperature_2m,precipitation_probability,soil_moisture_0_to_1cm,snowfall,dew_point_2m,relative_humidity_2m,evapotranspiration,precipitation,snow_depth,soil_temperature_0cm",
            "timezone": self.settings["timezone"],
            "past_days": 3,
        }

        try:
            logger.info(f"Fetching real-time weather data for {self.latitude}, {self.longitude}")
            response = requests.get(self.api_url, params=params)
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

    @staticmethod
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


    def process_data(self, json_data: dict) -> Optional[dict]:
        """Process API response into features for real-time weather data."""
        if not json_data:
            return None

        try:
            if not all(k in json_data for k in ["hourly", "daily"]):
                logger.warning("API response missing required sections (hourly or daily)")
                return None

            hourly_data = json_data["hourly"]
            daily_data = json_data["daily"]

        # Arrondir le temps actuel à l'heure pile (ex: 06:00)
            now = datetime.utcnow()
            rounded_time = now.replace(minute=0, second=0, microsecond=0)
            current_time = rounded_time.strftime("%Y-%m-%dT%H:%M")
            hourly_times = hourly_data.get("time", [])

        # Trouver l'index de l'heure actuelle ou la plus proche
            try:
                time_index = hourly_times.index(current_time)
            except ValueError:
                # Chercher l'heure la plus proche si l'exacte n'existe pas
                min_diff = float('inf')
                best_index = None
                for i, t in enumerate(hourly_times):
                    try:
                        t_obj = datetime.strptime(t, "%Y-%m-%dT%H:%M")
                        diff = abs((t_obj - rounded_time).total_seconds())
                        if diff < min_diff:
                            min_diff = diff
                            best_index = i
                    except ValueError:
                        continue
                if best_index is not None:
                    time_index = best_index
                    current_time = hourly_times[time_index]
                else:
                    logger.warning("No valid time found in hourly data")
                    return None

        # Récupérer les précipitations journalières
            daily_dates = daily_data.get("time", [])
            precip_sum_1d = daily_data.get("precipitation_sum", [])

            if len(daily_dates) != len(precip_sum_1d):
                logger.warning(f"Mismatch between daily dates and precipitation values: {len(daily_dates)} vs {len(precip_sum_1d)}")

            today_date = datetime.utcnow().strftime("%Y-%m-%d")
            if today_date in daily_dates:
                precipitation_value = precip_sum_1d[daily_dates.index(today_date)]
                precipitation_date = today_date
            else:
            # Utiliser la date la plus récente
                most_recent_date = max(daily_dates, key=lambda date: datetime.strptime(date, "%Y-%m-%d"))
                precipitation_value = precip_sum_1d[daily_dates.index(most_recent_date)]
                precipitation_date = most_recent_date

        # Construction des features météo (hourly variables are fetched exactly as before)
            features = {
                "current_time": current_time,
                "precipitation_date": precipitation_date,
                "precip_sum_1d": precipitation_value,
                "temperature_2m": self.safe_extract(hourly_data, ["temperature_2m", time_index]),
                "precipitation_probability": self.safe_extract(hourly_data, ["precipitation_probability", time_index]),
                "evapotranspiration": self.safe_extract(hourly_data, ["evapotranspiration", time_index]),
                "snowfall": self.safe_extract(hourly_data, ["snowfall", time_index]),
                "precipitation": self.safe_extract(hourly_data, ["precipitation", time_index]),
                "wind_gusts_10m": self.safe_extract(hourly_data, ["wind_gusts_10m", time_index]),
                "pressure_msl": self.safe_extract(hourly_data, ["pressure_msl", time_index]),
                "dewpoint_2m": self.safe_extract(hourly_data, ["dew_point_2m", time_index]),
                "soil_moisture": self.safe_extract(hourly_data, ["soil_moisture_0_to_7cm", time_index]),
                "humidity_2m": self.safe_extract(hourly_data, ["relative_humidity_2m", time_index]),
                "wind_speed_10m": self.safe_extract(hourly_data, ["wind_speed_10m", time_index]),
                "cloud_cover": self.safe_extract(hourly_data, ["cloud_cover", time_index]),
                "snow_depth": self.safe_extract(hourly_data, ["snow_depth", time_index]),
                "soil_temp": self.safe_extract(hourly_data, ["soil_temperature_0_to_7cm", time_index]),
            }

            return features

        except Exception as e:
            logger.error(f"Error processing real-time weather data: {str(e)}", exc_info=True)
            return None

    def normalize_realtime_weather_data(self):
        """Normalise les données météorologiques"""
        try:
            df = pd.read_csv(self.output_file)

            scaler = MinMaxScaler()
            df = df.drop(columns=["current_time"], axis=1, errors='ignore')
            # Colonnes à normaliser
            columns_to_normalize = [
                "temperature_2m", "precipitation_probability", "soil_moisture", 
                "snowfall", "dewpoint_2m", "humidity_2m", "evapotranspiration", 
                "precipitation", "snow_depth", "soil_temp","precip_sum_1d",
            ]
            
            # Appliquer la normalisation sur les colonnes numériques
            df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
            logger.info("Real-time weather data normalized.")
            return df
        except Exception as e:
            logger.error(f"Error normalizing weather data: {e}")
            return None

    def clean_realtime_weather_data(self):
        """Nettoie et prépare les données météorologiques"""

        column_names = [
            "current_time", "precipitation_date", "precip_sum_1d", "temperature_2m", 
            "precipitation_probability", "evapotranspiration", "snowfall", "precipitation", 
            "wind_gusts_10m", "pressure_msl", "dewpoint_2m", "soil_moisture", "humidity_2m", 
            "wind_speed_10m", "cloud_cover", "snow_depth", "soil_temp"
        ]
        try:
            df = pd.read_csv(self.output_file, header=None, names=column_names, on_bad_lines='skip')

            df['temperature_2m'] = df['temperature_2m'].fillna(df['temperature_2m'].mean())
            df['precipitation_probability'] = df['precipitation_probability'].fillna(0)
            df['soil_moisture'] = df['soil_moisture'].fillna(df['soil_moisture'].mean())
            df['snowfall'] = df['snowfall'].fillna(0)
            df['dewpoint_2m'] = df['dewpoint_2m'].fillna(df['dewpoint_2m'].mean())
            df['humidity_2m'] = df['humidity_2m'].fillna(df['humidity_2m'].mean())
            df['evapotranspiration'] = df['evapotranspiration'].fillna(0)
            df['precipitation'] = df['precipitation'].fillna(0)
            df['snow_depth'] = df['snow_depth'].fillna(0)
            df['soil_temp'] = df['soil_temp'].fillna(df['soil_temp'].mean())
            
            
            
            df = df.dropna()

            # Optionnel: Traiter les valeurs aberrantes
            df = df[(df["temperature_2m"] >= -50) & (df["temperature_2m"] <= 50)]
            df = df[(df["precipitation"] >= 0) & (df["precipitation"] <= 500)]

            logger.info("Real-time weather data cleaned.")
            return df
        except Exception as e:
            logger.error(f"Error cleaning weather data: {e}")
            return None

    def get_and_process_weather_data(self) -> Optional[pd.DataFrame]:
        """
        Récupère et traite les données météo en temps réel.
        """
        weather_data = self.fetch_data()

        if weather_data:
            processed_data = self.process_data(weather_data)
            if processed_data:
                # Save the processed data to a CSV file
                pd.DataFrame([processed_data]).to_csv(self.output_file, mode='a', header=True, index=False)
                
                return processed_data
        return None
