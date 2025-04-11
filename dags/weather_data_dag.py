from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
import logging
import json
import pandas as pd
from historic_data import fetch_flood_weather_data, clean_weather_data, combine_results
from realtime_data import OpenMeteoRealTimeFetcher

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
weather_fetcher = OpenMeteoRealTimeFetcher()

# Chemin vers votre fichier de métadonnées
SCENE_METADATA_PATH = '/opt/airflow/data/SEN12FLOOD/S1list.json'
HISTORIC_WEATHER_PATH = '/opt/airflow/data/store/combined_historic_weather_features.csv'
REALTIME_WEATHER_PATH = '/opt/airflow/data/store/weather_realtime_data.csv'

# Fonction pour charger les métadonnées
def load_scene_metadata():
    try:
        with open(SCENE_METADATA_PATH, 'r') as f:
            scene_metadata = json.load(f)
            logger.info(f"Chargé {len(scene_metadata)} scènes depuis {SCENE_METADATA_PATH}")
            return scene_metadata
    except Exception as e:
        logger.error(f"Erreur lors du chargement des métadonnées : {e}")
        return {}

# Fonction principale pour récupérer les données météorologiques
def fetch_weather_data_from_scenes():
    scene_metadata = load_scene_metadata()
    if scene_metadata:
        # Appel à votre fonction qui traite les données météorologiques
        df = fetch_flood_weather_data(scene_metadata)
        if not df.empty:
            logger.info(f"Données météorologiques récupérées avec succès. Nombre de lignes : {len(df)}")
            # Sauvegarder les données dans un fichier ou une base de données ici
            df.to_csv(HISTORIC_WEATHER_PATH, index=False)
        else:
            logger.warning("Aucune donnée météorologique traitée.")
    else:
        logger.warning("Aucune métadonnée de scène disponible.")

# Fonction pour combiner les fichiers de données en un seul
def combine_weather_data():
    try:
        # Lire les fichiers de données historiques et en temps réel
        historic_data = pd.read_csv(HISTORIC_WEATHER_PATH, on_bad_lines='skip')
        realtime_data = pd.read_csv(REALTIME_WEATHER_PATH, on_bad_lines='skip')
        realtime_data['flood_label'] = 0
        # Trouver les colonnes manquantes dans chaque DataFrame
        missing_columns_historic = set(realtime_data.columns) - set(historic_data.columns)
        missing_columns_realtime = set(historic_data.columns) - set(realtime_data.columns)

        # Ajouter des colonnes manquantes avec des valeurs 0 dans historic_data
        for col in missing_columns_historic:
            historic_data[col] = 0

        # Ajouter des colonnes manquantes avec des valeurs 0 dans realtime_data
        for col in missing_columns_realtime:
            realtime_data[col] = 0

        # Assurez-vous que les colonnes sont dans le même ordre avant de concaténer
        historic_data = historic_data[realtime_data.columns]

        # Ajouter les données en temps réel en dessous des données historiques (concatenation verticale)
        combined_data = pd.concat([historic_data, realtime_data], ignore_index=True)

        # Remplir les valeurs manquantes restantes par 0 (au cas où)
        combined_data.fillna(0, inplace=True)

        # Sauvegarder les données combinées
        combined_data.to_csv('/opt/airflow/data/store/combined_historic_realtime_weather_data.csv', index=False)
        logger.info("Données combinées sauvegardées avec succès.")
        
    except Exception as e:
        logger.error(f"Erreur lors de la combinaison des données : {e}")


# Définir les arguments par défaut pour le DAG
default_args = {
    'owner': 'airflow',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 4, 8),  # Ajustez la date de démarrage
    'catchup': False,  # Ne pas exécuter les tâches passées
}

# Créer un DAG
dag = DAG(
    'weather_data_fetcher',
    default_args=default_args,
    description='DAG pour récupérer et traiter les données météorologiques pour les scènes de inondation',
    schedule="@hourly",  # Exécution quotidienne, ajustez selon votre besoin
    max_active_runs=1,
)
create_folder_task = BashOperator(
    task_id='create_folder',
    bash_command='mkdir -p /opt/airflow/data/store',
    dag=dag,    )



# Définir une tâche Python pour récupérer et traiter les données météorologiques (historique)
fetch_weather_data_task = PythonOperator(
    task_id='fetch_weather_data_from_scenes',
    python_callable=fetch_weather_data_from_scenes,
    dag=dag,
)
combiner_weather_data_task = PythonOperator(
    task_id='combine_weather_data_from_scenes',
    python_callable=combine_results,
    dag=dag,
)
combiner_weather_realtime_historic_data_task = PythonOperator(
    task_id='combine_weather_realtime_historic_data_from_scenes',
    python_callable=combine_weather_data,
    dag=dag,
)

# Définir une tâche Python pour récupérer et traiter les données météorologiques en temps réel
get_realtime_weather_data_task = PythonOperator(
    task_id='get_and_process_weather_data',
    python_callable=weather_fetcher.get_and_process_weather_data,
    dag=dag,
)

# Tâches de nettoyage des données météo (historique et en temps réel)
clean_realtime_weather_data_task = PythonOperator(
    task_id='clean_realtime_weather_data',
    python_callable=weather_fetcher.clean_realtime_weather_data,
    dag=dag,
)
trigger_fourth_dag = TriggerDagRunOperator(
        task_id='trigger_trainig_processing_dag',
        trigger_dag_id='flood_prediction_dag',  # The ID of the DAG to trigger
        conf={"message": "third DAG has completed!"},  # Optional configuration
    )
clean_weather_data_task = PythonOperator(
    task_id='clean_weather_data',
    python_callable=clean_weather_data,
    dag=dag,
)


create_folder_task>>[fetch_weather_data_task, get_realtime_weather_data_task ] >> combiner_weather_data_task >> [clean_weather_data_task, clean_realtime_weather_data_task] >>combiner_weather_realtime_historic_data_task >> trigger_fourth_dag
