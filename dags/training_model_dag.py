import os
from airflow import DAG
from airflow.models import Variable

from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Define the default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 4, 10),
    'retries': 1,
}

def create_predict_directory():
    # Check if the 'predict' directory exists, and create it if not
    predict_dir = '/opt/airflow/data/predict'
    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir)
        print(f"Directory '{predict_dir}' created.")
    else:
        print(f"Directory '{predict_dir}' already exists.")

def preprocess_data():
    # Load the dataset
    dataset = pd.read_csv("/opt/airflow/data/store/combined_historic_realtime_weather_data.csv")
    
    # Drop unnecessary columns
    dataset = dataset.drop(columns=['timestamp'], errors='ignore')
    dataset = dataset.drop(columns=['current_time'], errors='ignore')
    dataset = dataset.drop(columns=['acquisition_date'], errors='ignore')
    dataset = dataset.drop(columns=['precipitation_date'], errors='ignore')
    dataset = dataset.drop(columns=['scene_id'], errors='ignore')
    dataset = dataset.drop(columns=['batch_number'], errors='ignore')
    dataset = dataset.drop(columns=['precip_sum_2d'], errors='ignore')
    dataset = dataset.drop(columns=['data_timestamp'], errors='ignore')
    dataset.to_csv('/opt/airflow/data/store/processed_weather_data.csv', index=False)
    
    # Feature selection
    X = dataset.drop(columns=['flood_label'], errors='ignore')
    y = dataset.get('flood_label')
    
    if y is None:
        raise ValueError("Target variable 'flood_label' is missing from the dataset.")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save the preprocessed data for the next steps
    joblib.dump(scaler, '/opt/airflow/data/predict/scaler.pkl')
    joblib.dump((X_train, X_test, y_train, y_test), '/opt/airflow/data/predict/preprocessed_data.pkl')

def reset_vars():
    Variable.set("dag1_status", "")
    Variable.set("dag2_status", "")

def train_model():
    # Load the preprocessed data
    X_train, X_test, y_train, y_test = joblib.load('/opt/airflow/data/predict/preprocessed_data.pkl')
    
    # Initialize and train the RandomForest model
    model = RandomForestClassifier(random_state=15)
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, "/opt/airflow/data/predict/flood_prediction_model.pkl")
    
    # Evaluate model accuracy
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

import joblib
import pandas as pd
import logging

def make_prediction(**kwargs):
    model = joblib.load("/opt/airflow/data/predict/flood_prediction_model.pkl")
    scaler = joblib.load('/opt/airflow/data/predict/scaler.pkl')

    input_dict = {
        "precip_sum_1d": [5.2],
        "temperature_2m": [18.3],
        "precipitation_probability": [75],
        "evapotranspiration": [2.1],
        "snowfall": [0],
        "precipitation": [4.8],
        "wind_gusts_10m": [32],
        "pressure_msl": [1013],
        "dewpoint_2m": [12.1],
        "soil_moisture": [0.3],
        "humidity_2m": [80],
        "wind_speed_10m": [10],
        "cloud_cover": [65],
        "snow_depth": [0],
        "soil_temp": [15.2],
        "relative_humidity_2m": [82],
        "soil_temperature_0_to_7cm": [14.8],
        "soil_moisture_0_to_7cm": [0.28],
        "dew_point_2m": [11.9]
    }

    
    input_df = pd.DataFrame(input_dict)
    input_data_scaled = scaler.transform(input_df)      
    prediction = model.predict(input_data_scaled)
   
    flood_status = "Flood" if prediction[0] == 1 else "No Flood"
    
    logging.info(f"Flood prediction result: {flood_status} (Prediction: {prediction[0]}) with data {input_dict}")
    print(f"Flood prediction result: {flood_status} (Prediction: {prediction[0]}) with data {input_dict}")  
    task_instance = kwargs.get('ti')
    task_instance.xcom_push(key='flood_status', value=flood_status)
    
    return flood_status


# Define the DAG
dag = DAG(
    'flood_prediction_dag',
    default_args=default_args,
    description='Flood Prediction Model Training and Prediction',
    schedule_interval=None,  
)

# Define the tasks
create_dir_task = PythonOperator(
    task_id='create_predict_directory',
    python_callable=create_predict_directory,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

reset_task = PythonOperator(
    task_id='reset_status_vars',
    python_callable=reset_vars
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

predict_task = PythonOperator(
    task_id='make_prediction',
    python_callable=make_prediction,
    dag=dag,
)

# Task dependencies
create_dir_task >> preprocess_task >> train_task  >> reset_task >> predict_task
