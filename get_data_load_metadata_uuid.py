# Copyright Â© 2024 Accenture. All rights reserved.

# File: get_data_load_metadata_uuid.py
# Author: chitresh.goyal
# Date: 19-Mar-2024
# Description: Get uuid reference from data_load_metadata table.

import scripts.Helper as Helper
import scripts.dbConfig as db
import scripts.config as config

# Database connection parameters
postgresPort = db.postgresPort
postgresHost = db.postgresHost
postgresDB = db.postgresDB
postgresUser = db.postgresUser
postgresPassword = db.postgresPassword

audience_analytics_table_name = config.audience_analytics_table_name
individual_3p_table_name = config.individual_3p_table_name
data_load_metadata_table_name = config.dataLoadMetadataTableName

connection_string = f'postgresql://{postgresUser}:{postgresPassword}@{postgresHost}:{postgresPort}/{postgresDB}'

def get_last_data_load_metadata_uuid():
    df = Helper.select_table_data(data_load_metadata_table_name, cols=['id', 'load_date'], connection_string=connection_string)
    df = df.sort_values(by='load_date', ascending=False).reset_index(drop=True)
    return df.iloc[0]['id'] if not df.empty else None
