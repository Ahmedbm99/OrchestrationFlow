from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
from airflow.models import Variable
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from skimage.feature import graycomatrix, graycoprops
from SEN12FloodDataset import SEN12FloodDataset, FloodFeatureExtractor, analyze_features  # Adjust the import as needed

def set_status_and_maybe_trigger(**context):
    # Set the status of DAG1
    Variable.set("dag2_status", "success")
    
    # Check if DAG2 status is "success"
    if Variable.get("dag1_status", default_var="") == "success":
        # Push True to XCom if the condition is met
        context['ti'].xcom_push(key='trigger', value=True)
    else:
        # Push False to XCom if the condition is not met
        context['ti'].xcom_push(key='trigger', value=False)

def extract_and_analyze_features(**context):
    # Initialize the dataset and feature extractor
    dataset = SEN12FloodDataset(img_size=256)  # You can adjust img_size or other parameters
    feature_extractor = FloodFeatureExtractor(dataset)

    # Extract features from the dataset (you can specify num_samples if needed)
    features_df = feature_extractor.extract_features()

    # Analyze the features and save the results
    analyze_features(features_df)

    # You can also return any results you need for downstream tasks (optional)
    return features_df

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='satellite_data_processing',
    default_args=default_args,
    description='DAG pour récupérer et traiter les données météorologiques pour les scènes de inondation',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
) as dag:

    check_and_trigger = PythonOperator(
        task_id='check_and_trigger_final',
        python_callable=set_status_and_maybe_trigger,
        provide_context=True,
    )

    trigger_final = TriggerDagRunOperator(
        task_id="trigger_final",
        trigger_dag_id="flood_prediction_dag",
        wait_for_completion=True,
        trigger_rule="all_done",
        execution_date="{{ ds }}",  # or `{{ execution_date }}` as needed
        reset_dag_run=True,
        poke_interval=30,
        deferrable=True,

    )

    # Extract and analyze features task
    extract_and_analyze_task = PythonOperator(
        task_id='extract_and_analyze_features',
        python_callable=extract_and_analyze_features,
        provide_context=True,
    )

    # Load and extract data from scenes task (if needed)
    load_satellite_data_task = PythonOperator(
        task_id='load_extract_data_from_scenes',
        python_callable=extract_and_analyze_features,  # Reuse the same function if applicable
    )

    # Set task dependencies
    load_satellite_data_task >> check_and_trigger >> trigger_final
    extract_and_analyze_task >> check_and_trigger
