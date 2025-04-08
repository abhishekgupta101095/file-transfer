from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
import airflow.utils.dates
from scripts.invokeIDR import invokeIDRProcess

default_args = {
        "owner": "airflow", 
        "start_date": airflow.utils.dates.days_ago(1)
    }

with DAG(dag_id="invokeIDR", default_args=default_args, schedule_interval=None) as dag:
    
    start = DummyOperator(
    task_id='start'
    )
    
    invokeIDR_task = PythonOperator(
    task_id='invokeIDR_task', 
    python_callable=invokeIDRProcess
    )
        
    end = DummyOperator(
    task_id='end'
    )

start >> invokeIDR_task >> end