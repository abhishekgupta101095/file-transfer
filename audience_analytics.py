# Copyright Â© 2024 Accenture. All rights reserved.

# File: audience_demo_insights.py
# Author: chitresh.goyal
# Date: 15-Mar-2024
# Description: Process and generate audience analytics view

import uuid
import datetime
import json
import pandas as pd
import scripts.Helper as Helper
import scripts.dbConfig as db
import scripts.config as config
from scripts.get_data_load_metadata_uuid import get_last_data_load_metadata_uuid

# Database connection parameters
postgresPort = db.postgresPort
postgresHost = db.postgresHost
postgresDB = db.postgresDB
postgresUser = db.postgresUser
postgresPassword = db.postgresPassword

audience_analytics_table_name = config.audience_analytics_table_name
individual_3p_table_name = config.individual_3p_table_name
data_load_metadata_table_name = config.dataLoadMetadataTableName
attribute_config = config.attribute_configs

connection_string = f'postgresql://{postgresUser}:{postgresPassword}@{postgresHost}:{postgresPort}/{postgresDB}'

# Load JSON data
with open(attribute_config) as f:
    json_data = json.load(f)

# Define global variables
global data_provider
global vendor_name
global dataset_count

data_provider = json_data['vendor_data'][0]['data_provider']
vendor_name = json_data['vendor_data'][0]['vendor_name']
dataset_count = json_data['vendor_data'][0]['dataset_count']


def insert_data_load_metadata():
    global data_provider
    global vendor_name
    load_date = datetime.datetime.now()
    loaded_by = ""  # Change to your name or the appropriate value

    # Create a DataFrame with the metadata
    metadata_df = pd.DataFrame({
        'id': [str(uuid.uuid4())],
        'load_date': [load_date],
        'loaded_by': [loaded_by],
        'data_provider': [data_provider],
        'data_vendor': [vendor_name]
    })

    # Insert the DataFrame into the database
    Helper.write_dataframe_to_postgresql_in_chunks(metadata_df, data_load_metadata_table_name, connection_string=connection_string)


def create_audience_analytics_view_process():
    global dataset_count
    df_3p_individual = Helper.select_table_data(individual_3p_table_name, cols=[], connection_string=connection_string)
    
    # Get the UUID of the last data load metadata entry and ensure it's in the correct format
    data_load_metadata_uuid = get_last_data_load_metadata_uuid()

    # Calculate the required values
    identities = len(df_3p_individual)
    devices = len(df_3p_individual['device_id'].unique())
    
    # Create a DataFrame for the audience_analytics table
    df_audience_analytics = pd.DataFrame({
        'data_load_metadata_id': [data_load_metadata_uuid],
        'identities': [identities],
        'devices': [devices],
        'datasets': [dataset_count],
        'attributes':[100]
    })
    
   
    # Write the DataFrame to the database
    Helper.write_dataframe_to_postgresql_in_chunks(df_audience_analytics, audience_analytics_table_name, chunksize=10000, connection_string=connection_string)

# if __name__ == '__main__':
#     insert_data_load_metadata()
#     create_audience_analytics_view_process()






