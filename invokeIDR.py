import requests
import json
from datetime import datetime

import scripts.config as config
import scripts.Helper as Helper
import scripts.dbConfig as db
import pandas as pd

container=config.containername_idrinvoke
database=config.database_idrinvoke
table=config.table_idrinvoke
columns=config.columns_idrinvoke
location=config.s3location_idrinvoke
url=config.url_idrinvoke
bucketname=config.bucketName_idrinvoke
access_key_id=config.aws_access_key_id_idrinvoke
secret_access_key=config.aws_secret_access_key_idrinvoke
job_status_table_name=config.job_status_table_name

postgresPort=db.postgresPort
postgresHost=db.postgresHost
postgresDB=db.postgresDB
postgresUser=db.postgresUser
postgresPassword=db.postgresPassword

# Database connection parameters
db_params = {
    'host': postgresHost,
    'port': postgresPort,
    'database': postgresDB,
    'user': postgresUser,
    'password': postgresPassword
}

connection_string=f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["database"]}'


def invokeIDRProcess():
    # Define notification payload
    current_date_time = datetime.now()
    request_id = current_date_time.strftime('%Y%m%d%H%M%S')
    payload = {
        "request_id": request_id,
        "input_data_info": {	    
    		"container":container,
    		"database":database,
    		"table":table,
    		"columns": columns
        },
    	"s3_details": {
    	    "bucketname":bucketname,
    		"location":location
    	}
    }

    headers = {"Content-Type": "application/json"}
    
    jobstatus_pg = pd.DataFrame({
            'request_id': [request_id],
            'job_name': ['invokeIDR'],
            'status': [''],
            'failure_reason': ['']})
            
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        
        # Handle response
        if response.status_code == 200:
            jobstatus_pg['status'] = 'success'
            print("Notification sent successfully.")
        else:
            jobstatus_pg['status'] = 'failure'
            jobstatus_pg['failure_reason'] = response.status_code
            raise ValueError(f"Failed to send notification. Status code: '{response.status_code}' ")
            
        Helper.write_dataframe_to_postgresql_in_chunks(jobstatus_pg,job_status_table_name, chunksize=10000, connection_string=connection_string)
    except Exception as e:
        jobstatus_pg['status'] = 'failure'
        jobstatus_pg['failure_reason'] = 'error while sending request to server'
        Helper.write_dataframe_to_postgresql_in_chunks(jobstatus_pg,job_status_table_name, chunksize=10000, connection_string=connection_string)
        raise ValueError(f"Failed to connect to make request. With error: '{e}' ")
