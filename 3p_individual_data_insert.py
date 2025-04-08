#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright Â© 2024 Accenture. All rights reserved.
# Author: chitresh.goyal
# Date: 18-April-2024
# Description: 3p_individual table data insertion script

import pandas as pd
from sqlalchemy import create_engine
import dbConfig as db
import config as config

# Specify the path to your CSV file
csv_file_path = '3p_individual.csv'

# Fetch the database connection parameters from dbConfig module
postgresPort = db.postgresPort
postgresHost = db.postgresHost
postgresDB = db.postgresDB
postgresUser = db.postgresUser
postgresPassword = db.postgresPassword

# Fetch the table name from the config module
individual_3p_table_name = config.individual_3p_table_name

# Construct the database connection string
connection_string = f'postgresql://{postgresUser}:{postgresPassword}@{postgresHost}:{postgresPort}/{postgresDB}'

try:
    # Create a database engine
    engine = create_engine(connection_string)
    print("Database connection established.")

    # Read the entire CSV with header
    df = pd.read_csv(csv_file_path)

    # Optional: Handle missing values or perform data transformation if necessary
    # Example: Fill missing 'cid' with a default value (remove if not applicable)
    df['cid'].fillna('default_cid_value', inplace=True)

    # Write the data to the PostgreSQL table
    df.to_sql(name=individual_3p_table_name, con=engine, if_exists='append', index=False)

    print("Data inserted successfully.")
except Exception as e:
    print(f"An error occurred: {e}")





