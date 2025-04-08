# -*- coding: utf-8 -*-

# Copyright Â© 2024 Accenture. All rights reserved.

# File: idr_profile_poi.py
# Author: shahnoor.alam
# Date: 18-April-2024
# Description: Process to generate resolved identities top visited places view.

import pandas as pd
import scripts.Helper as Helper
import scripts.dbConfig as db
import ast
import scripts.config as config
from scripts.get_data_load_metadata_uuid import get_last_data_load_metadata_uuid

# Database connection parameters
postgresPort = db.postgresPort
postgresHost = db.postgresHost
postgresDB = db.postgresDB
postgresUser = db.postgresUser
postgresPassword = db.postgresPassword

poiDictCatTableName=config.poiDictCatTableName
individual_3p_table_name=config.individual_3p_table_name
idr_profile_poi_table_name=config.idr_profile_poi_table_name
persistentId = "persistentId"

connection_string = f'postgresql://{postgresUser}:{postgresPassword}@{postgresHost}:{postgresPort}/{postgresDB}'

def idr_profile_poi_view_process():
    
    df_poi_category = Helper.select_table_data(poiDictCatTableName, cols=[], connection_string=connection_string)
    df_3p_original = Helper.select_table_data(individual_3p_table_name, cols=[], connection_string=connection_string)
    df_3p_individual = df_3p_original[df_3p_original[persistentId].notna()]
    
    df_individual = df_3p_individual.filter(['cid', 'visits'], axis=1)
    df_individual['visits'] = [ast.literal_eval(visits_str.replace('# Visits to ', '')) for visits_str in df_individual['visits']]
    
    # Convert the dictionary into a list of dictionaries
    list_of_dicts = []
    for cid, visits_dict in zip(df_individual['cid'], df_individual['visits']):
        for key, value in visits_dict.items():
            list_of_dicts.append({'cid': cid, 'visits': key})
            
    # Create DataFrame from the list of dictionaries
    df_3p_individual_poi = pd.DataFrame(list_of_dicts)
    df_3p_individual_poi.rename(columns={'visits': 'matching_data'}, inplace=True)
    
    
    # Preprocessing for merging
    df_3p_individual_poi['matching_data'] = df_3p_individual_poi['matching_data'].str.strip().str.lower()
    df_poi_category['location_lower'] = df_poi_category['location'].str.strip().str.lower()
    
    new_location = df_poi_category[['category', 'subcategory', 'location', 'location_lower']]
    merged_df = pd.merge(df_3p_individual_poi, new_location, left_on='matching_data', right_on='location_lower', how='left', indicator=True)
    merged_df.drop(columns=['location_lower'], inplace=True)  # Drop the lowercase column

    columns_to_remove = ['cid', 'matching_data', '_merge']
    merged_df = merged_df.drop(columns_to_remove, axis=1)
    merged_df = merged_df.drop_duplicates()
    
    # Calculate percentages
    total_rows = len(df_3p_individual)
    
    # Get the UUID of the last data load metadata entry and ensure it's in the correct format
    data_load_metadata_uuid = get_last_data_load_metadata_uuid()

    # Create the location_visited view
    location_visited = pd.DataFrame({
        'top_category': merged_df['category'],
        'sub_category': merged_df['subcategory'],
        'location_name': merged_df['location'],
        'tot_pop': len(df_3p_individual),
        'data_load_metadata_id': [data_load_metadata_uuid] * len(merged_df)  # Add the metadata UUID as a column

    })
    
    # Calculate and assign percentages
    location_visited['percentage_top_category'] = location_visited.groupby('top_category')['top_category'].transform(
        lambda x: (x.count() / total_rows) * 100
    )
    location_visited['sub_category_count'] = location_visited.groupby(['top_category', 'sub_category'])['sub_category'].transform('count')
    location_visited['percentage_sub_category'] = location_visited.groupby(['top_category', 'sub_category'])['sub_category'].transform(
        lambda x: (x.count() / location_visited.groupby('top_category')['top_category'].transform('count')) * 100
    )
    
    location_visited = location_visited.drop_duplicates()
    
    Helper.write_dataframe_to_postgresql_in_chunks(location_visited, idr_profile_poi_table_name, chunksize=10000, connection_string=connection_string)
# if __name__ == '__main__':
#     idr_profile_poi_view_process()