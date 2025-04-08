# -*- coding: utf-8 -*-

# Copyright © 2024 Accenture. All rights reserved.

# File: idr_web_view.py
# Author: chitresh.goyal
# Date: 17-April-2024
# Description: Process and generate idr web view.

# import Helper
# import dbConfig as db
# import config as config

import pandas as pd
import ast
import scripts.Helper as Helper
import scripts.dbConfig as db
import scripts.config as config

postgresPort = db.postgresPort
postgresHost = db.postgresHost
postgresDB = db.postgresDB
postgresUser = db.postgresUser
postgresPassword = db.postgresPassword

idr_web_view_table_name = config.idr_web_view_table_name
individual_3p_table_name = config.individual_3p_table_name
webDictCatTableName = config.webDictCatTableName

# Database connection parameters
db_params = {
    'host': postgresHost,
    'port': postgresPort,
    'database': postgresDB,
    'user': postgresUser,
    'password': postgresPassword
}

connection_string = f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["database"]}'

def idr_web_view_process():
    df_web_category = Helper.select_table_data(webDictCatTableName, cols=[], connection_string=connection_string)
    df_web_category['category_Tier1'] = df_web_category['category'].str.cat(df_web_category['tier1'], sep='_')
    df_web_category['category_Tier1_Tier2'] = df_web_category['category_Tier1'].str.cat(df_web_category['tier2'], sep='_')

    df_3p_original = Helper.select_table_data(individual_3p_table_name, cols=[], connection_string=connection_string)
    df_3p_original = df_3p_original[df_3p_original['persistentId'].notna()]  # Apply persistentID not null check
    df_individual = df_3p_original.filter(['cid', 'num_domain_surf'], axis=1)
    df_individual['num_domain_surf'] = [ast.literal_eval(num_domain_surf_str.replace('Num_DomainSurf_', '')) for num_domain_surf_str in df_individual['num_domain_surf']]

    list_of_dicts = []
    for cid, num_domain_surf_dict in zip(df_individual['cid'], df_individual['num_domain_surf']):
        for key, value in num_domain_surf_dict.items():
            list_of_dicts.append({'cid': cid, 'tier2': key})

    df_3p_individual = pd.DataFrame(list_of_dicts)
    df_3p_individual.rename(columns={'tier2': 'matching_data'}, inplace=True)

    new_tier2 = df_web_category[['category', 'tier1', 'tier2']]
    new_category_Tier1 = df_web_category[['category', 'tier1', 'tier2', 'category_Tier1']]
    new_category = df_web_category[['category', 'tier1', 'tier2']]
    new_category_Tier1_Tier2 = df_web_category[['category', 'tier1', 'tier2', 'category_Tier1', 'category_Tier1_Tier2']]

    merged_df = pd.merge(df_3p_individual, new_tier2, left_on='matching_data', right_on='tier2', how='left', indicator=True)
    merged_df = merged_df[merged_df['tier2'].notna()]

    merged1_df = pd.merge(df_3p_individual, new_category_Tier1, left_on='matching_data', right_on='category_Tier1', how='left', indicator=True)
    merged1_df = merged1_df[merged1_df['category_Tier1'].notna()]

    merged2_df = pd.merge(df_3p_individual, new_category, left_on='matching_data', right_on='category', how='left', indicator=True)
    merged2_df = merged2_df[merged2_df['category'].notna()]

    merged3_df = pd.merge(df_3p_individual, new_category_Tier1_Tier2, left_on='matching_data', right_on='category_Tier1_Tier2', how='left', indicator=True)
    merged3_df = merged3_df[merged3_df['category_Tier1_Tier2'].notna()]

    result = pd.concat([merged_df, merged1_df, merged2_df, merged3_df], ignore_index=True)
    result = result.drop_duplicates()
    columns_to_remove = ['cid', '_merge', 'category_Tier1', 'category_Tier1_Tier2', 'matching_data']
    result = result.drop(columns_to_remove, axis=1)
    result = result.drop_duplicates()

    full = pd.DataFrame({
        'category': result['category'],
        'tier1': result['tier1'],
        'tier2': result['tier2'],
        'num_ind_cat': result.groupby(['category'])['category'].transform('count'),
        'num_ind_tier1': result.groupby(['category', 'tier1'])['category'].transform('count'),
        'num_ind_tier2': result.groupby(['category', 'tier1', 'tier2'])['category'].transform('count'),
        'num_ind_tier2_benchmark': result.groupby(['category', 'tier1', 'tier2'])['category'].transform('count'),
        'num_ind_tier1_benchmark': result.groupby(['category', 'tier1'])['tier1'].transform('count'),
        'num_ind_cat_benchmark': result.groupby(['category'])['category'].transform('count'),
        'tot_pop': df_3p_original.shape[0],
        'perc_ind_cat': "",
        'perc_ind_tier1': "",
        'perc_ind_tier2': ""
    })

    full = full.drop_duplicates()
    full['perc_ind_cat'] = full['num_ind_cat'] / full['tot_pop']
    full['perc_ind_tier1'] = full['num_ind_tier1'] / full['tot_pop']
    full['perc_ind_tier2'] = full['num_ind_tier2'] / full['tot_pop']

    result = full.copy()

    Helper.write_dataframe_to_postgresql_in_chunks(result, idr_web_view_table_name, chunksize=10000, connection_string=connection_string)

# if __name__ == '__main__':
#     idr_web_view_process()
