from airflow import DAG
from airflow.providers.amazon.aws.operators.s3 import S3ListOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.python import BranchPythonOperator
from airflow.sensors.external_task_sensor import ExternalTaskSensor
import airflow.utils.dates

from datetime import datetime, timedelta
import os
from pathlib import Path
import boto3

import scripts.config as config
from scripts.idr_demo_insights import idr_demo_insights_view_process
from scripts.idr_profile_app import idr_profile_app_view_process
from scripts.idr_web_view import idr_web_view_process
from scripts.idr_social_view import idr_social_view_process
from scripts.idr_profile_poi import idr_profile_poi_view_process



BUCKET_NAME=config.bucketName
METADATA_LOCAL_PATH=config.locatPathToDownloadDictionaryFiles
S3_DICTIONARY_PREFIX=config.s3DictionaryPrefix
S3_DELIMITER=config.s3Delimiter
AWS_CONN_ID=config.awsConnId

BUCKET_NAME=config.bucketName
RAWDATA_LOCAL_PATH=config.locatPathToDownloadRawFiles
S3_RAW_FILES_PREFIX=config.s3RawFilesPrefix
S3_DELIMITER=config.s3Delimiter
AWS_CONN_ID=config.awsConnId
aws_access_key_id=config.aws_access_key_id
aws_secret_access_key=config.aws_secret_access_key
aws_region=config.aws_region
s3_processed_path=config.s3_processed_path

default_args = {
        "owner": "airflow", 
        "start_date": airflow.utils.dates.days_ago(1)
    }


with DAG(dag_id="createIDR3PViews", default_args=default_args, schedule_interval=None) as dag:    
    start = DummyOperator(
    task_id='start'
    )
       
    idr_demo_insights_view = PythonOperator(
        task_id="idr_demo_insights_view",
        python_callable=idr_demo_insights_view_process
    )
    
    idr_profile_app_view = PythonOperator(
        task_id="idr_profile_app_view",
        python_callable=idr_profile_app_view_process
    )
    
    idr_web_view = PythonOperator(
        task_id="idr_web_view",
        python_callable=idr_web_view_process
    )
    
    idr_social_view = PythonOperator(
        task_id="idr_social_view",
        python_callable=idr_social_view_process
    )
    
    idr_profile_poi_view = PythonOperator(
        task_id="idr_profile_poi_view",
        python_callable=idr_profile_poi_view_process
    )
        
    squash = DummyOperator(
    task_id='squash'
    )
    
    end = DummyOperator(
    task_id='end'
    )


start >> [idr_demo_insights_view,idr_profile_app_view,idr_web_view,idr_social_view,idr_profile_poi_view] >> squash >> end

