from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime
#from docker.types import Mount  # Add this import

default_args = {
    'owner': 'Ebrahim Emad',
    'start_date': datetime(2025, 7, 11),
}

with DAG(
    dag_id='ml_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
    tags=['ml', 'pipeline'],
) as dag:
    dataset_creation_task = BashOperator(
        task_id="faked_dataset_creation_task",
        bash_command="""
        echo "Hey the dataset is ready, let's trigger the training process"
        """
    )

    training_task = BashOperator(
        task_id='training_task',
        bash_command="python /opt/airflow/scripts/code.py",
    )

    dataset_creation_task >> training_task