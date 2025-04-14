from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
from airflow.models import Variable
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from SEN12FloodDataset import (
    SEN12FloodDataset,
    FloodFeatureExtractor,
    analyze_features
)  # Adjust if needed

def set_status_and_maybe_trigger(**kwargs):
    # Set the status of DAG2 to success
    Variable.set("dag2_status", "success")

    # Check the status of DAG1
    dag1_status = Variable.get("dag1_status", default_var="")
    should_trigger = dag1_status == "success"

    # Push decision to XCom
    kwargs['ti'].xcom_push(key='trigger', value=should_trigger)

def extract_and_analyze_features():
    dataset = SEN12FloodDataset(img_size=256)
    feature_extractor = FloodFeatureExtractor(dataset)
    features_df = feature_extractor.extract_features()
    analyze_features(features_df)
    return True

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='satellite_data_processing',
    default_args=default_args,
    description='DAG pour récupérer et traiter les données satellites pour les scènes d’inondation',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
) as dag:

    load_satellite_data_task = PythonOperator(
        task_id='load_extract_data_from_scenes',
        python_callable=extract_and_analyze_features,
    )

    extract_and_analyze_task = PythonOperator(
        task_id='extract_and_analyze_features',
        python_callable=extract_and_analyze_features,
    )

    check_and_trigger = PythonOperator(
        task_id='check_and_trigger_final',
        python_callable=set_status_and_maybe_trigger,
    )

    trigger_final = TriggerDagRunOperator(
        task_id="trigger_final",
        trigger_dag_id="flood_prediction_dag",
        wait_for_completion=True,
        trigger_rule="all_done",
        execution_date="{{ ds }}",
        reset_dag_run=True,
        poke_interval=15,
        deferrable=True,
    )

    # Task dependencies
    [load_satellite_data_task, extract_and_analyze_task] >> check_and_trigger >> trigger_final
