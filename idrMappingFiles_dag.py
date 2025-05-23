from airflow import DAG
from airflow.providers.amazon.aws.operators.s3 import S3ListOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.microsoft.azure.hooks.wasb import WasbHook
from datetime import datetime, timedelta
import airflow.utils.dates

import os
from pathlib import Path

from airflow.utils.trigger_rule import TriggerRule

import scripts.config as config

import boto3
from airflow.operators.python import BranchPythonOperator

from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from scripts.idr_mapping_files import getGoldenRecordIDProcess

BUCKET_NAME=config.idr_mappingfile_bucketName
IDRMAPPING_DATA_LOCAL_PATH=config.localPathToDownloadIDRMappingFiles
S3_IDRMAPPING_PREFIX=config.idr_mappingFileS3Prefix
S3_DELIMITER=config.s3Delimiter
AWS_CONN_ID=config.awsConnId

S3_RAW_FILES_PREFIX=config.s3RawFilesPrefix
aws_access_key_id=config.aws_access_key_id
aws_secret_access_key=config.aws_secret_access_key
aws_region=config.aws_region
s3_processed_path=config.s3_processed_path_idr_mapping

default_args = {
        "owner": "airflow", 
        "start_date": airflow.utils.dates.days_ago(1)
    }

    
def download_from_s3(key: str, s3_prefix: str, localFilePath: str) -> None:
    hook = S3Hook('aws_udm')
    if key == s3_prefix:
        return None
    file_name = hook.download_file(
        key=key,
        bucket_name=BUCKET_NAME,
        preserve_file_name=True,
        local_path=localFilePath,
        use_autogenerated_subdir=False
    )
    # will return absolute path
    return file_name


def list_files_and_locations(directory_path):
    file_list = []
    
    # Iterate over files in the specified directory
    for root, dirs, files in os.walk(directory_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            #file_list.append({'file_name': file_name, 'file_path': file_path})
            file_list.append(file_path)
    
    return file_list
    


def move_s3_objects(key: str, s3_prefix: str) -> None:
    if key == s3_prefix:
        return None
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )
    current_date = datetime.now()
    formatted_date = current_date.strftime('%Y%m%d')
    dirPath = key.rsplit('/', 1)[0]
    fileName = key.rsplit('/', 1)[1]
    newKey = dirPath + '/' + formatted_date + '/' + fileName
    destination_key = newKey.replace(S3_IDRMAPPING_PREFIX, s3_processed_path, 1)
    # Copy the object to the new location
    s3.copy_object(
            Bucket=BUCKET_NAME,
            CopySource={'Bucket': BUCKET_NAME, 'Key': key},
            Key=destination_key
        )



def delete_s3_objects(key: str, s3_prefix: str) -> None:
    if key == s3_prefix:
        return None
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )
    s3.delete_object(Bucket=BUCKET_NAME, Key=key)


def delete_local_files(filePath: str):
    # Get a list of all files in the directory
    file_list = os.listdir(filePath)
    # Iterate through the list and delete each file
    for file_name in file_list:
        file_path = os.path.join(filePath, file_name)
        try:
            # Delete the file
            os.remove(file_path)
            print(f"File {file_name} has been successfully deleted.")
        except Exception as e:
            print(f"An error occurred while deleting {file_name}: {e}")


with DAG(dag_id="getGoldenRecordID", default_args=default_args, schedule_interval=None) as dag:
    
    start = DummyOperator(
    task_id='start'
    )
    
    listIDRMappingFilesS3 = S3ListOperator(
        task_id="listIDRMappingFilesS3",
        bucket=BUCKET_NAME,
        prefix=S3_IDRMAPPING_PREFIX,
        delimiter=S3_DELIMITER,
        aws_conn_id=AWS_CONN_ID
    )
    
    delete_local_IDRMapping_files = PythonOperator(
        task_id='delete_local_IDRMapping_files',
        python_callable=delete_local_files,
        op_kwargs={
            'filePath': IDRMAPPING_DATA_LOCAL_PATH
        }
    )
    
    download_idrmapping_files_from_s3 = PythonOperator.partial(
        task_id='download_idrmapping_files_from_s3',
        python_callable=download_from_s3
        ).expand(
          op_kwargs=listIDRMappingFilesS3.output.map(
            lambda x: {
                "key":f"{x}",
                "s3_prefix":S3_IDRMAPPING_PREFIX,
                "localFilePath":IDRMAPPING_DATA_LOCAL_PATH
            }
        )
    )
        
    listIDRMappingFilesLocal = PythonOperator(
        task_id='list_files_task',
        python_callable=list_files_and_locations,
        op_args=[IDRMAPPING_DATA_LOCAL_PATH],
        provide_context=True
    )

    idrMapping_processing = PythonOperator.partial(
        task_id='idrMapping_processing',
        python_callable=getGoldenRecordIDProcess
        ).expand(
          op_kwargs=listIDRMappingFilesLocal.output.map(
            lambda x: {
                "filePath":f"{x}"
            }
        )
    )
    
    delete_local_IDRMapping_files_final = PythonOperator(
        task_id='delete_local_IDRMapping_files_final',
        python_callable=delete_local_files,
        op_kwargs={
            'filePath': IDRMAPPING_DATA_LOCAL_PATH
        }
    )
    
    move_files_task = PythonOperator.partial(
        task_id='move_files_task',
        python_callable=move_s3_objects
        ).expand(
          op_kwargs=listIDRMappingFilesS3.output.map(
            lambda x: {
                "key":f"{x}",
                "s3_prefix":S3_IDRMAPPING_PREFIX
            }
        )
    )
    
    delete_s3_objects = PythonOperator.partial(
        task_id='delete_s3_objects',
        python_callable=delete_s3_objects
        ).expand(
          op_kwargs=listIDRMappingFilesS3.output.map(
            lambda x: {
                "key":f"{x}",
                "s3_prefix":S3_IDRMAPPING_PREFIX
            }
        )
    )
    
    end = DummyOperator(
    task_id='end'
    )

start >> listIDRMappingFilesS3 >> delete_local_IDRMapping_files >> download_idrmapping_files_from_s3  >> listIDRMappingFilesLocal >> idrMapping_processing >> delete_local_IDRMapping_files_final >>  move_files_task >> delete_s3_objects >> end
