from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

with DAG(
    'install_packages_dag',
    default_args=default_args,
    description='A DAG to install necessary Python packages',
    schedule="@daily",  # Runs daily at midnight
    start_date=datetime(2025, 4, 1),  # Ensure this date is in the past
    catchup=False,  # Don't run past missed DAGs

) as dag:

    # Task to trigger weather_data_fetcher DAG
    trigger_first_dag = TriggerDagRunOperator(
        task_id='trigger_weather_data_fetcher',
        trigger_dag_id='weather_data_fetcher',  # The ID of the DAG to trigger
        conf={"message": "First DAG has completed!"},  # Optional configuration
    )

    # Task to trigger SEN12Flood_Feature_Extraction DAG
    trigger_second_dag = TriggerDagRunOperator(
        task_id='trigger_satellite_data_processing_dag',
        trigger_dag_id='satellite_data_processing_dag',  # The ID of the DAG to trigger
        conf={"message": "First DAG has completed!"},  # Optional configuration
    )

    # Task to upgrade pip
    upgrade_packages = BashOperator(
        task_id='upgrade_packages',
        bash_command='pip install --upgrade pip',
    )

    # Task to install required Python packages
    install_packages = BashOperator(
        task_id='install_packages',
        bash_command='pip install seaborn torch scikit-learn tqdm scikit-image matplotlib rasterio opencv-python ',
    )

    # Task dependencies: upgrade first, then install packages, and finally trigger other DAGs
    upgrade_packages >> install_packages >> [trigger_first_dag, trigger_second_dag]
