from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import requests
import pandas as pd
import os
import json
from datetime import datetime, timedelta
import logging


logging.basicConfig(level=logging.INFO)

DATA_DIR = "/opt/airflow/data"
STATIC_DATA_PATH = os.path.join(DATA_DIR, "weather-type-classification.zip")
STORMGLASS_API_KEY = "bf83ec50-e995-11ef-acf2-0242ac130003-bf83ecf0-e995-11ef-acf2-0242ac130003"
TOMORROW_IO_API_KEY = "aPoVXe703EmFbhYILvBAaXDKwLTSi3wq"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def save_path_database(path):
    """Save the file path to the PostgreSQL database."""
    hook = PostgresHook(postgres_conn_id="postgres_default")
    conn = hook.get_conn()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_data (
            id SERIAL PRIMARY KEY,
            file_path TEXT,
            timestamp TIMESTAMP DEFAULT NOW()
        );
    """)

    cursor.execute("""INSERT INTO weather_data (file_path) VALUES (%s);""", (path,))
    conn.commit()
    conn.close()

    logging.info(f"Path {path} stored in PostgreSQL.")

def fetch_weather_data():
    """Fetch meteorological data and save it to CSV."""
    try:
        response = requests.get(
            'https://api.stormglass.io/v2/weather/point',
            params={
                'lat': 34.7398,
                'lng': 10.7600,
                'params': 'airTemperature,humidity,windSpeed,pressure,precipitation,waveHeight,wavePeriod,cloudCover,seaLevel,soilMoisture',
                'start': start,
                'end': end
            },
            headers={'Authorization': STORMGLASS_API_KEY}
        )

        if response.status_code == 200:
            json_data = response.json()
            if "hours" in json_data:
                df = pd.DataFrame(json_data["hours"])
                file_path = os.path.join(DATA_DIR, "Weather_data.csv")
                df.to_csv(file_path, index=False)
                logging.info(f"Weather data saved to {file_path}")
                
                # Save the file path to PostgreSQL
                save_path_database(file_path)
            else:
                logging.error("Empty or malformed weather data.")
        else:
            logging.error(f"Failed to fetch weather data: {response.status_code} {response.text}")
            response.raise_for_status()
    except Exception as e:
        logging.error(f"Error in fetch_weather_data: {e}")
        raise



def fetch_realtime_weather_data():
    """Fetch real-time weather data using Tomorrow.io API."""
    url = f"https://api.tomorrow.io/v4/weather/realtime?location=sfax&apikey={TOMORROW_IO_API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            realtime_data = response.json()
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            realtime_path = os.path.join(DATA_DIR, f"realtime_weather_data_{timestamp}.json")
            with open(realtime_path, "w") as f:
                json.dump(realtime_data, f)
            logging.info(f"Real-time weather data saved at {realtime_path}.")
            return realtime_path
        else:
            logging.error(f"Failed to fetch real-time weather data: {response.status_code}")
            raise Exception("API request failed.")
    except Exception as e:
        logging.error(f"Error in fetch_realtime_weather_data: {e}")
        raise

def merge_weather_data(ti):
    """Merge static, real-time, and StormGlass weather data."""
    realtime_path = ti.xcom_pull(task_ids="fetch_realtime_weather_data")
    stormglass_path = os.path.join(DATA_DIR, "Weather_data.csv")  # Static path for StormGlass output

    static_data = pd.read_csv(STATIC_DATA_PATH, compression="zip")

    with open(realtime_path, "r") as f:
        realtime_data = json.load(f)
    realtime_df = pd.json_normalize(realtime_data)

    if os.path.exists(stormglass_path):
        stormglass_data = pd.read_csv(stormglass_path)
    else:
        logging.warning(f"StormGlass data file not found at {stormglass_path}.")
        stormglass_data = pd.DataFrame()

    merged_data = pd.concat([static_data, realtime_df, stormglass_data], axis=0, ignore_index=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    merged_path = os.path.join(DATA_DIR, f"merged_weather_data_{timestamp}.csv")
    merged_data.to_csv(merged_path, index=False)

    logging.info(f"Merged weather data saved at {merged_path}.")
    return merged_path

def clean_weather_data(ti):
    merged_path = ti.xcom_pull(task_ids="merge_weather_data")
    merged_data = pd.read_csv(merged_path)

    numeric_cols = merged_data.select_dtypes(include="number").columns
    if not numeric_cols.empty:
        merged_data[numeric_cols] = (
            (merged_data[numeric_cols] - merged_data[numeric_cols].mean()) /
            merged_data[numeric_cols].std()
        )

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    cleaned_path = os.path.join(DATA_DIR, f"cleaned_weather_data_{timestamp}.csv")
    merged_data.dropna().to_csv(cleaned_path, index=False)
    logging.info(f"Cleaned weather data saved at {cleaned_path}.")
    return cleaned_path

def publish_weather_data(ti):
    """Save the cleaned data file path to PostgreSQL."""
    cleaned_path = ti.xcom_pull(task_ids="clean_weather_data")
    save_path_database(cleaned_path)

with DAG(
    dag_id="weather_data_processing",
    schedule=(timedelta(hours=3)),
    start_date=datetime(2025, 2, 14),
    catchup=False,
) as dag_weather:


    download_static_dataset = PythonOperator(
        task_id="download_static_dataset",
        python_callable=lambda: os.system(f"kaggle datasets download nikhil7280/weather-type-classification -p {DATA_DIR}"),
    )

    fetch_realtime_data = PythonOperator(
        task_id="fetch_realtime_weather_data",
        python_callable=fetch_realtime_weather_data,
    )
    fetch_weather_task = PythonOperator(
    task_id="fetch_weather_data",
    python_callable=fetch_weather_data,
    )

    merge_data = PythonOperator(
        task_id="merge_weather_data",
        python_callable=merge_weather_data,
    )

    clean_data = PythonOperator(
        task_id="clean_weather_data",
        python_callable=clean_weather_data,
    )

    publish_data = PythonOperator(
        task_id="publish_weather_data",
        python_callable=publish_weather_data,
    )

    # Task dependencies
    [download_static_dataset, fetch_realtime_data, fetch_weather_task] >> merge_data >> clean_data >> publish_data 

