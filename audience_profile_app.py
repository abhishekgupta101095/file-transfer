# Copyright Â© 2024 Accenture. All rights reserved.

# File: audience_profile_poi.py
# Author: chitresh.goyal
# Date: 15-Mar-2024
# Description: Process and generate audience mobile app view.

# -*- coding: utf-8 -*-

import pandas as pd
import scripts.Helper as Helper
import scripts.dbConfig as db
import ast
import scripts.config as config
from scripts.get_data_load_metadata_uuid import get_last_data_load_metadata_uuid

postgresPort = db.postgresPort
postgresHost = db.postgresHost
postgresDB = db.postgresDB
postgresUser = db.postgresUser
postgresPassword = db.postgresPassword

audience_profile_app_table_name = config.audience_profile_app_table_name
individual_3p_table_name = config.individual_3p_table_name

# Database connection parameters
db_params = {
    'host': postgresHost,
    'port': postgresPort,
    'database': postgresDB,
    'user': postgresUser,
    'password': postgresPassword
}

connection_string = f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["database"]}'

def audience_profile_app_view_process():
    df_apps_data = Helper.select_table_data(individual_3p_table_name, cols=[], connection_string=connection_string)
    df_apps_transform = df_apps_data.filter(['cid', 'no_of_apps'], axis=1)
    
    # Get the UUID of the last data load metadata entry and ensure it's in the correct format
    data_load_metadata_uuid = get_last_data_load_metadata_uuid()

    df_app_view = []
    for _, row in df_apps_transform.iterrows():
        cid = row['cid']
        apps = ast.literal_eval(row['no_of_apps'])
        for category, app_list in apps.items():
            for app_name in app_list:
                df_app_view.append({'cid': cid, 'category': category.split('_')[-1], 'app_name': app_name})

    df_app_view = pd.DataFrame(df_app_view)

    df_app_view_full = pd.DataFrame({
        'category': df_app_view['category'],
        'app_name': df_app_view['app_name'],
        'num_ind': df_app_view.groupby(['category', 'app_name'])['category'].transform('count'),
        'tot_pop': df_apps_data.shape[0],
        'perc_ind': '',
        'data_load_metadata_id': [data_load_metadata_uuid] * len(df_app_view)  # Add the metadata UUID as a column
    })

    df_app_view_full['perc_ind'] = df_app_view_full['num_ind'] / df_app_view_full['tot_pop']
    df_app_view_full = df_app_view_full.drop_duplicates()

    Helper.write_dataframe_to_postgresql_in_chunks(df_app_view_full, audience_profile_app_table_name, chunksize=10000, connection_string=connection_string)

# if __name__ == '__main__':
#     audience_profile_app_view_process()
