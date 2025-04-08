# -*- coding: utf-8 -*-

# Copyright Â© 2024 Accenture. All rights reserved.

# File: idr_social_view.py
# Author: chitresh.goyal
# Date: 18-April-2024
# Description: Process and generate idr social view.

import pandas as pd
# import Helper
# import dbConfig as db
# import config as config

import scripts.Helper as Helper
import scripts.dbConfig as db
import ast
import scripts.config as config
from ast import literal_eval

postgresPort = db.postgresPort
postgresHost = db.postgresHost
postgresDB = db.postgresDB
postgresUser = db.postgresUser
postgresPassword = db.postgresPassword

segment_profile_poi_table_name = config.segment_profile_poi_table_name
poiDictCatTableName = config.poiDictCatTableName
individual_3p_table_name = config.individual_3p_table_name
segment_profile_app_table_name = config.segment_profile_app_table_name
idr_social_view_table_name = config.idr_social_view_table_name
social_meta_table_name = config.social_meta_table_name

# Database connection parameters
db_params = {
    'host': postgresHost,
    'port': postgresPort,
    'database': postgresDB,
    'user': postgresUser,
    'password': postgresPassword
}

connection_string = f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["database"]}'

# Function to convert list items to rows
def explode_social(row):
    social_list = row['social']
    if isinstance(social_list, str):
        social_list = social_list.replace('nan,', "'brands_travel,'")
        social_list = social_list.replace(" nan", "'brands_travel,'")
        social_list = literal_eval(social_list)
    elif not isinstance(social_list, list):
        social_list = [social_list]  # Convert non-list values to list
    return [(row['cid'], social) for social in social_list]

def idr_social_view_process():
    df_social_category = Helper.select_table_data(social_meta_table_name, cols=[], connection_string=connection_string)
    df_social_category['cat1_cat2'] = df_social_category['category1'].str.cat(df_social_category['category2'], sep='_')
    df_social_category['cat1_cat2_cat3'] = df_social_category['cat1_cat2'].str.cat(df_social_category['category3'], sep='_')
    df_social_category['cat1_cat2_Cat3_cat6'] = df_social_category['cat1_cat2_cat3'].str.cat(df_social_category['category6'], sep='_')

    df_original = Helper.select_table_data(individual_3p_table_name, cols=[], connection_string=connection_string)
    df_original = df_original[df_original['persistentId'].notna()]  # Apply persistentId not null check
    df_individual = df_original[['cid', 'social']]
    rows = df_individual.apply(explode_social, axis=1).explode()

    df_3p_individual = pd.DataFrame(rows.tolist(), columns=['cid', 'matching_data'])
    new_cat1_cat2 = df_social_category[['category1', 'category2', 'category3', 'category6', 'cat1_cat2']]
    new_cat1_cat2_Cat3_cat6 = df_social_category[['category1', 'category2', 'category3', 'category6', 'cat1_cat2_Cat3_cat6']]

    merged_df = pd.merge(df_3p_individual, new_cat1_cat2_Cat3_cat6, left_on='matching_data', right_on='cat1_cat2_Cat3_cat6', how='left', indicator=True)
    merged_df = merged_df.drop_duplicates()

    filtered_new_cat1_cat2 = new_cat1_cat2[new_cat1_cat2['category3'].isnull()]
    merged2_df = pd.merge(df_3p_individual, filtered_new_cat1_cat2, left_on='matching_data', right_on='cat1_cat2', how='left', indicator=True)
    merged2_df = merged2_df[merged2_df['cat1_cat2'].notna()]
    merged2_df = merged2_df.drop_duplicates()

    result = pd.concat([merged_df, merged2_df], ignore_index=True)
    filtered_result = result[result['category1'].notna()]
    result = result.drop_duplicates()

    columns_to_remove = ['cid', '_merge', 'cat1_cat2_Cat3_cat6', 'cat1_cat2', 'matching_data']
    filtered_result = filtered_result.drop(columns_to_remove, axis=1)
    filtered_result = filtered_result.drop_duplicates()

    full = pd.DataFrame({
        'cat1': filtered_result['category1'],
        'cat2': filtered_result['category2'],
        'cat3': filtered_result['category3'],
        'cat6': filtered_result['category6'],
        'num_ind_cat1': filtered_result.groupby(['category1'])['category1'].transform('count'),
        'num_ind_cat2': filtered_result.groupby(['category1', 'category2'])['category1'].transform('count'),
        'num_ind_cat3': filtered_result.groupby(['category1', 'category2', 'category3'])['category1'].transform('count'),
        'num_ind_cat6': filtered_result.groupby(['category1', 'category2', 'category3', 'category6'])['category1'].transform('count'),
        'tot_pop': df_original.shape[0],
        'perc_ind_cat1': "",
        'perc_ind_cat2': "",
        'perc_ind_cat3': "",
        'perc_ind_cat6': ""
    })

    full['perc_ind_cat1'] = full['num_ind_cat1'] / full['tot_pop']
    full['perc_ind_cat2'] = full['num_ind_cat2'] / full['tot_pop']
    full['perc_ind_cat3'] = full['num_ind_cat3'] / full['tot_pop']
    full['perc_ind_cat6'] = full['num_ind_cat6'] / full['tot_pop']

    Helper.write_dataframe_to_postgresql_in_chunks(full, idr_social_view_table_name, chunksize=10000, connection_string=connection_string)

#if __name__ == '__main__':
#    idr_social_view_process()