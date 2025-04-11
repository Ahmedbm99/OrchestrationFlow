# /opt/airflow/dags/flood_feature_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta


# Import the main function
from SEN12FloodDataset import main

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='satellite_data_processing',
    default_args=default_args,
    description='DAG pour récupérer et traiter les données météorologiques pour les scènes de inondation',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
   
) as dag:

    load_satellite_data_task = PythonOperator(
        task_id='load_extract_data_from_scenes',
        python_callable=main,
    )

    load_satellite_data_task
