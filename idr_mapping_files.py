import glob
import pandas as pd
import psycopg2
from sqlalchemy import create_engine

import logging

import scripts.dbConfig as db
import scripts.config as config
import re


table_name=config.individual_3p_table_name
cidCol=config.cidCol
persistentIDCol=config.persistentIDCol

postgresConnection=db.postgresConnection
postgresDriver=db.postgresDriver
postgresUser=db.postgresUser
postgresPassword=db.postgresPassword
postgresPort=db.postgresPort
postgresHost=db.postgresHost
postgresDB=db.postgresDB

# Database connection parameters
db_params = {
    'host': postgresHost,
    'port': postgresPort,
    'database': postgresDB,
    'user': postgresUser,
    'password': postgresPassword
}

connection_string=f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["database"]}'


def getGoldenRecordIDProcess(filePath):
    files = glob.glob(filePath)
    
    # Create a SQLAlchemy engine for PostgreSQL
    engine = create_engine(f'postgresql+psycopg2://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["database"]}')

    for file_path in files:
        # Read data from the CSV file using pandas
        logging.info("Reading file",file_path)
        idrMappingFileDF = pd.read_csv(file_path,encoding='utf-8')
        
        idrMappingFileDF = idrMappingFileDF[[cidCol,persistentIDCol]]
        
        
        # Iterate over DataFrame and execute update statements
        for index, row in idrMappingFileDF.iterrows():
            update_query = f"UPDATE \"{table_name}\" SET {persistentIDCol} = '{row[persistentIDCol]}' WHERE {cidCol} = '{row[cidCol]}'"
            with engine.connect() as connection:
                connection.execute(update_query)
        
        print("Bulk update completed.")
 