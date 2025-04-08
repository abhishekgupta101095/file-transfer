import glob
import pandas as pd
import psycopg2
from sqlalchemy import create_engine

import logging

import scripts.dbConfig as db
import scripts.config as config
import re

localPath=config.locatPathToDownloadDictionaryFiles
filePatternToLoad=config.filePatternToLoadDictionaryFiles
webDictCols=config.webDictCols
webCatTableName=config.webDictCatTableName
appsDictCols=config.appsDictCols
appsDictCatTableName=config.appsDictCatTableName
poiDictCols=config.poiDictCols
poiDictCatTableName=config.poiDictCatTableName
socialDictCols=config.socialDictCols
social_meta_table_name=config.social_meta_table_name

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


def insert_into_table(filePath: str,engine, data):
    logging.info("Inserting into a postgres table")
    # Insert data into the table
    apps_match = re.search('apps', filePath)
    web_match = re.search('web', filePath)
    poi_match = re.search('poi', filePath)
    social_match = re.search('social', filePath)
    
    if apps_match:
        columns=appsDictCols
        table_name=appsDictCatTableName
        
    if web_match:
        columns=webDictCols
        table_name=webCatTableName
        
    if poi_match:
        columns=poiDictCols
        table_name=poiDictCatTableName
    
    if social_match:
        columns=socialDictCols
        table_name=social_meta_table_name

    logging.info('Table name is: ',table_name)
    logging.info('Columns are: ',' '.join([str(elem) for elem in columns]))
    
    # Construct the SQL query to truncate the table
    truncate_query = f'TRUNCATE TABLE \"{table_name}\" RESTART IDENTITY;'
    
    # Execute the truncate query
    with engine.connect() as connection:
        connection.execute(truncate_query)
    
    df = pd.DataFrame(data, columns=columns)
    logging.info('Dataframe is : ',df)
    df.to_sql(table_name, engine, if_exists='append', index=False)


def process_s3(filePath):
    # Create a SQLAlchemy engine for PostgreSQL
    engine = create_engine(f'postgresql+psycopg2://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["database"]}')

    # Find CSV files based on the pattern
    files = glob.glob(filePath)

    for file_path in files:
        # Read data from the CSV file using pandas
        logging.info("Reading file",file_path)
        df = pd.read_csv(file_path,encoding='latin1')

        # Insert data into the PostgreSQL table
        insert_into_table(filePath,engine, df.values)