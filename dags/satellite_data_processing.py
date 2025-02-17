from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import ftplib
import logging
import pandas as pd
from datetime import datetime, timedelta
import tarfile
import matplotlib.pyplot as plt

DATA_DIR = "/opt/airflow/data"

TABLE_CREATION_QUERY = """
CREATE TABLE IF NOT EXISTS processed_data (
    id SERIAL PRIMARY KEY,
    file_path TEXT NOT NULL,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

def save_to_database(file_path):
    """Save file path and timestamp into the database."""
    hook = PostgresHook(postgres_conn_id="postgres_default")
    conn = hook.get_conn()
    cursor = conn.cursor()
    cursor.execute(TABLE_CREATION_QUERY)
    conn.commit()
    
    cursor.execute("INSERT INTO processed_data (file_path) VALUES (%s)", (file_path,))
    conn.commit()
    conn.close()
    logging.info(f"File path {file_path} saved to the database.")

def fetch_ghcnh_data():
    """Download GHCNh data from NOAA's FTP server."""
    try:
        ftp = ftplib.FTP("ftp.ncdc.noaa.gov")
        ftp.login("anonymous", "benmakhloufahmed1@gmail.com")  
        ftp.cwd("/pub/data/ghcn/daily")  

        ghcnh_dir = os.path.join(DATA_DIR, "ghcnh")
        os.makedirs(ghcnh_dir, exist_ok=True)

        file_name = f"ghcnh_data_{datetime.now().strftime('%Y%m%d%H%M%S')}.tar.gz"
        file_path = os.path.join(ghcnh_dir, file_name)

        with open(file_path, "wb") as f:
            ftp.retrbinary("RETR ghcnd_hcn.tar.gz", f.write)

        ftp.quit()
        logging.info(f"GHCNh data downloaded successfully to {file_path}.")
        return file_path
    except ftplib.all_errors as e:
        logging.error(f"FTP error occurred: {e}")
        raise
    except Exception as e:
        logging.error(f"Error downloading GHCNh data: {e}")
        raise

def eda_on_data(ti):
    clean_path = ti.xcom_pull(task_ids="clean_ghcnh_data")
    data = pd.read_csv(clean_path)

    if "temperature" not in data.columns:
        raise ValueError("Column 'temperature' not found in the data.")

    summary = data.describe()
    logging.info(f"Data summary: \n{summary}")

    plt.hist(data["temperature"], bins=20)
    plt.title("Temperature Distribution")
    plt.savefig(os.path.join(DATA_DIR, "eda_temperature_histogram.png"))
    plt.close()

def preprocess_ghcnh_data(ti):
    """Preprocess and extract GHCNh data."""
    tar_path = ti.xcom_pull(task_ids="fetch_ghcnh_data")
    extract_dir = os.path.join(DATA_DIR, "ghcnh_extracted")
    os.makedirs(extract_dir, exist_ok=True)

    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            extracted_files = tar.getnames()
            logging.info(f"Extracted files: {extracted_files}")
            if not extracted_files:
                raise ValueError("The archive is empty.")
            tar.extractall(path=extract_dir)

        csv_files = [f for f in extracted_files if f.endswith(".csv")]
        if not csv_files:
            logging.warning("No CSV files found in the archive.")
            return None
        ghcnh_data_path = os.path.join(extract_dir, csv_files[0])

        logging.info(f"GHCNh data extracted to {ghcnh_data_path}.")
        return ghcnh_data_path
    except tarfile.TarError as e:
        logging.error(f"Error extracting the archive: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during extraction: {e}")
        raise

def clean_ghcnh_data(ti):
    """Clean and normalize GHCNh data."""
    ghcnh_path = ti.xcom_pull(task_ids="preprocess_ghcnh_data")
    clean_dir = os.path.join(DATA_DIR, "ghcnh_cleaned")
    os.makedirs(clean_dir, exist_ok=True)

    ghcnh_data = pd.read_csv(ghcnh_path)

    required_columns = ["temperature", "date"]
    for col in required_columns:
        if col not in ghcnh_data.columns:
            raise ValueError(f"Required column '{col}' not found in the data.")

    initial_rows = ghcnh_data.shape[0]
    ghcnh_data.dropna(inplace=True)
    ghcnh_data = ghcnh_data.rename(columns=str.lower)
    ghcnh_data["timestamp"] = pd.to_datetime(ghcnh_data["date"])

    file_name = f"ghcnh_cleaned_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    clean_path = os.path.join(clean_dir, file_name)
    ghcnh_data.to_csv(clean_path, index=False)
    logging.info(f"Cleaned GHCNh data saved at {clean_path}.")
    logging.info(f"Data shape after cleaning: {ghcnh_data.shape}")
    logging.info(f"Number of rows dropped: {initial_rows - ghcnh_data.shape[0]}")
    return clean_path

def publish_ghcnh_data(ti):
    """Publish cleaned GHCNh data and save file path to the database."""
    clean_path = ti.xcom_pull(task_ids="clean_ghcnh_data")
    save_to_database(clean_path)  
    logging.info(f"Published and logged cleaned data from {clean_path}.")

with DAG(
    dag_id="ghcnh_data_processing",
    schedule=timedelta(hours=3), 
    start_date=datetime(2025, 2, 14),
    catchup=False,
) as dag_ghcnh:

    fetch_ghcnh = PythonOperator(
        task_id="fetch_ghcnh_data",
        python_callable=fetch_ghcnh_data,
    )

    preprocess_ghcnh = PythonOperator(
        task_id="preprocess_ghcnh_data",
        python_callable=preprocess_ghcnh_data,
    )

    clean_ghcnh = PythonOperator(
        task_id="clean_ghcnh_data",
        python_callable=clean_ghcnh_data,
    )

    visualization_ghcnh = PythonOperator(
        task_id="eda_on_data",
        python_callable=eda_on_data,
    )

    publish_ghcnh = PythonOperator(
        task_id="publish_ghcnh_data",
        python_callable=publish_ghcnh_data,
    )

    fetch_ghcnh >> preprocess_ghcnh >> clean_ghcnh >> visualization_ghcnh >> publish_ghcnh