# Copyright © 2024 Accenture. All rights reserved.

# File: idr_demo_insights.py
# Author: chitresh.goyal
# Date: 12-April-2024
# Description: Process and generate idr demographic insights view.

import pandas as pd
# import Helper
# import dbConfig as db
# import config as config
# from get_data_load_metadata_uuid import get_last_data_load_metadata_uuid

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

idr_demo_insights_table_name = config.idr_demo_insights_table_name
individual_3p_table_name=config.individual_3p_table_name
persistentId = "persistentId"

connection_string = f'postgresql://{postgresUser}:{postgresPassword}@{postgresHost}:{postgresPort}/{postgresDB}'

def custom_list_parser(string):
    # Check if the string starts and ends with square brackets
    if (string.startswith('{') and string.endswith('}')) or (string.startswith('[') and string.endswith(']')):
        # Remove the brackets and split the string by commas
        elements = string[1:-1].split(',')
        # Strip whitespace from each element and return as a list
        return [element.strip().strip("'") for element in elements]
    else:
        # Return the original string if it doesn't match the expected format
        return string.strip("'")

def process_data_view_for_attribute(attributename, tot_pop):
    df_3p_individual = Helper.select_table_data(individual_3p_table_name, cols=[], connection_string=connection_string)
    df_3p_individual = df_3p_individual[df_3p_individual[persistentId].notna()]
    
    # Get the UUID of the last data load metadata entry and ensure it's in the correct format
    data_load_metadata_uuid = get_last_data_load_metadata_uuid()
    
    if attributename == 'affinity':
        print(attributename)
        df_affinity = df_3p_individual.copy()
        df_affinity['affinity'] = df_affinity['affinity'].apply(custom_list_parser)
        df_affinity = df_affinity.explode('affinity')
        df_affinities = df_affinity['affinity'].value_counts().reset_index()
        df_affinities.columns = ['sortdata', 'record_count']
        df_affinities['sortcolumn'] = 'affinity'        
        df_affinities['rounded_percentage'] = ((df_affinities['record_count'] / df_affinities['record_count'].sum()) * 100).round(2)
        df_affinities['rank'] = df_affinities['rounded_percentage'].rank(ascending=False).astype(int)
        df_affinities['flag'] = 'I'  # Index value
        df_affinities['tot_pop'] = tot_pop  # Total population
        df_affinities['data_load_metadata_id'] = [data_load_metadata_uuid] * len(df_affinities)  # Add the metadata UUID as a column
        df_affinities['perc_ind'] = ((df_affinities['record_count'] / df_affinities['tot_pop']) * 100)
        df_affinities = df_affinities[['sortcolumn', 'sortdata', 'record_count', 'rounded_percentage', 'rank', 'flag', 'tot_pop', 'data_load_metadata_id', 'perc_ind']]
        return df_affinities
    else:
        df = df_3p_individual[attributename].value_counts().reset_index()
        df.columns = ['sortdata', 'record_count']
        df['rounded_percentage'] = ((df['record_count'] / df['record_count'].sum()) * 100).round(2)
        df['sortcolumn'] = attributename
        df['rank'] = df['rounded_percentage'].rank(ascending=False).astype(int)
        df['flag'] = 'P'  # Percentage value
        df['tot_pop'] = tot_pop  # Total population
        df['data_load_metadata_id'] = [data_load_metadata_uuid] * len(df)  # Add the metadata UUID as a column
        df['perc_ind'] = ((df['record_count'] / df['tot_pop']) * 100)
        df = df[['sortcolumn', 'sortdata', 'record_count', 'rounded_percentage', 'rank', 'flag', 'tot_pop', 'data_load_metadata_id', 'perc_ind']]
        return df

def idr_demo_insights_view_process():
    # Fetch the entire dataset from the individual_3p table
    df_3p_individual = Helper.select_table_data(individual_3p_table_name, cols=[], connection_string=connection_string)

    # Filter the DataFrame to only include rows where 'persistentId' is not null
    df_3p_individual = df_3p_individual[df_3p_individual[persistentId].notna()]

    tot_pop = len(df_3p_individual)
    
    attributes = ['married', 'no_of_adults_hh', 'no_of_children_hh', 'premium_card_holder', 'collar_worker', 'urbanicity', 'occupation', 'age_group', 'education_level', 'gender', 'home_owner_status', 'household_income', 'individual_income', 'net_worth', 'affinity']
    dfs = [process_data_view_for_attribute(attr, tot_pop) for attr in attributes]
    result = pd.concat(dfs, ignore_index=True)
    Helper.write_dataframe_to_postgresql_in_chunks(result, idr_demo_insights_table_name, chunksize=10000, connection_string=connection_string)

# if __name__ == '__main__':
#     idr_demo_insights_view_process()
