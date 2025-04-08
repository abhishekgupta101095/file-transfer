# Copyright © 2024 Accenture. All rights reserved.

# File: idr_demo_summary.py
# Author: abhishek.cw.gupta
# Date: 23-April-2024
# Description: Process and generate idr demographic insights summary view.

# import pandas as pd
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
idr_demo_summary_table_name = config.idr_demo_summary_table_name

connection_string = f'postgresql://{postgresUser}:{postgresPassword}@{postgresHost}:{postgresPort}/{postgresDB}'

def idr_demo_summary_view_process():

    #Required attribute
    require_fetch = ['married','no_of_children_hh','gender',"age_group","household_income","net_worth","education_level",'occupation','community','no_of_adults_hh']
    
    # Fetch the dataset from the idr_demo_insights table
    df_demo_insights = Helper.select_table_data(idr_demo_insights_table_name, cols=[], connection_string=connection_string)
    df_demo_insights_summary_t = df_demo_insights[['sortcolumn','record_count']]
     
    df_demo_insights_summary = df_demo_insights_summary_t.groupby(['sortcolumn'])['record_count'].sum()
    df_demo_insights_summary = df_demo_insights_summary.reset_index()

    df_demo_insights_summary['tot_pop'] = df_demo_insights['tot_pop'][0]

    df_demo_insights_summary.loc[df_demo_insights_summary['sortcolumn'] == 'urbanicity', 'sortcolumn'] = 'community'
    
    df_demo_insights_summary['rounded_percentage'] = ((df_demo_insights_summary['record_count']/df_demo_insights_summary['tot_pop'])*100).round(2)
    df_demo_insights_summary = df_demo_insights_summary[df_demo_insights_summary['sortcolumn'].isin(require_fetch)].reset_index()
    df_demo_insights_summary = df_demo_insights_summary.drop(['index'],axis=1)
    
    Helper.write_dataframe_to_postgresql_in_chunks(df_demo_insights_summary, idr_demo_summary_table_name, chunksize=10000, connection_string=connection_string)

#if __name__ == '__main__':
#   idr_demo_summary_view_process()
