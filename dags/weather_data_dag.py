from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.models import Variable
import logging
import json
import pandas as pd
from historic_data import fetch_flood_weather_data, clean_weather_data
from realtime_data import OpenMeteoRealTimeFetcher

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
weather_fetcher = OpenMeteoRealTimeFetcher()

SCENE_METADATA_PATH = '/opt/airflow/data/S1list.json'
HISTORIC_WEATHER_PATH = '/opt/airflow/data/store/combined_historic_weather_features.csv'
REALTIME_WEATHER_PATH = '/opt/airflow/data/store/weather_realtime_data.csv'
def load_scene_metadata():
    try:
        with open(SCENE_METADATA_PATH, 'r') as f:
            scene_metadata = json.load(f)
            logger.info(f"Chargé {len(scene_metadata)} scènes depuis {SCENE_METADATA_PATH}")
            return scene_metadata
    except Exception as e:
        logger.error(f"Erreur lors du chargement des métadonnées : {e}")
        return {}

def fetch_weather_data_from_scenes():
    fetch_flood_weather_data(
        metadata_file=SCENE_METADATA_PATH,
        output_file=HISTORIC_WEATHER_PATH
    )

def set_status_and_maybe_trigger(**context):
    # Set the status of DAG1
    Variable.set("dag1_status", "success")
    
    # Check if DAG2 status is "success"
    if Variable.get("dag2_status", default_var="") == "success":
        # Push True to XCom if the condition is met
        context['ti'].xcom_push(key='trigger', value=True)
    else:
        # Push False to XCom if the condition is not met
        context['ti'].xcom_push(key='trigger', value=False)


def combine_weather_data():
    try:
        historic_data = pd.read_csv(HISTORIC_WEATHER_PATH, on_bad_lines='skip')
        realtime_data = pd.read_csv(REALTIME_WEATHER_PATH, on_bad_lines='skip')
        realtime_data['flood_label'] = 0
        missing_columns_historic = set(realtime_data.columns) - set(historic_data.columns)
        missing_columns_realtime = set(historic_data.columns) - set(realtime_data.columns) 
        for col in missing_columns_historic:
            historic_data[col] = 0
        for col in missing_columns_realtime:
            realtime_data[col] = 0   
        historic_data = historic_data[realtime_data.columns]    
        combined_data = pd.concat([historic_data, realtime_data], ignore_index=True)     
        combined_data.fillna(0, inplace=True)
        combined_data.to_csv('/opt/airflow/data/store/combined_historic_realtime_weather_data.csv', index=False)
        logger.info("Données combinées sauvegardées avec succès.")
        
    except Exception as e:
        logger.error(f"Erreur lors de la combinaison des données : {e}")




default_args = {
    'owner': 'airflow',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 4, 8),  
    'catchup': False,  
}
dag = DAG(
    'weather_data_fetcher',
    default_args=default_args,
    description='DAG pour récupérer et traiter les données météorologiques pour les scènes de inondation',
    schedule=None, 
    max_active_runs=1,
)
create_folder_task = BashOperator(
    task_id='create_folder',
    bash_command='mkdir -p /opt/airflow/data/store',
    dag=dag,    )
fetch_weather_data_task = PythonOperator(
    task_id='fetch_weather_data_from_scenes',
    python_callable=fetch_weather_data_from_scenes,
    dag=dag,
)
clean_historic_weather_data_task = PythonOperator(
    task_id='clean_historic_weather_data',
    python_callable=lambda: clean_weather_data().to_csv(HISTORIC_WEATHER_PATH, index=False),
    dag=dag,
)
check_and_trigger = PythonOperator(
    task_id='check_and_trigger_final',
    python_callable=set_status_and_maybe_trigger,
    provide_context=True,
)

combiner_weather_realtime_historic_data_task = PythonOperator(
    task_id='combine_weather_realtime_historic_data_from_scenes',
    python_callable=combine_weather_data,
    dag=dag,
)
get_realtime_weather_data_task = PythonOperator(
    task_id='get_and_process_weather_data',
    python_callable=weather_fetcher.get_and_process_weather_data,
    dag=dag,
)
clean_realtime_weather_data_task = PythonOperator(
    task_id='clean_realtime_weather_data',
    python_callable=weather_fetcher.clean_realtime_weather_data,
    dag=dag,
)
trigger_fourth_dag = TriggerDagRunOperator(
    task_id="trigger_fourth_dag",
    trigger_dag_id="flood_prediction_dag",
    wait_for_completion=True,
    trigger_rule="all_done",
    execution_date="{{ ds }}",  # ou `{{ execution_date }}` selon besoin
    reset_dag_run=True,
    poke_interval=30,
    deferrable=True,
)



create_folder_task >> fetch_weather_data_task >> clean_historic_weather_data_task
create_folder_task >> get_realtime_weather_data_task >> clean_realtime_weather_data_task

[clean_historic_weather_data_task, clean_realtime_weather_data_task] >> combiner_weather_realtime_historic_data_task
combiner_weather_realtime_historic_data_task >> check_and_trigger >> trigger_fourth_dag