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

def make_prediction(input_data):
    # Load the trained model and scaler
    model = joblib.load("/opt/airflow/data/predict/flood_prediction_model.pkl")
    scaler = joblib.load('/opt/airflow/data/predict/scaler.pkl')
    
    # Preprocess the input data (using the same scaler as used for training)
    input_data_scaled = scaler.transform(input_data)
    
    # Make a prediction
    prediction = model.predict(input_data_scaled)
    return prediction

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
