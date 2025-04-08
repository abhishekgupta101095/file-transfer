'''
Create segment view for segment_profile_app
'''

import pandas as pd
import ast
import scripts.Helper as Helper
import scripts.dbConfig as db
import scripts.config as config
from scripts.get_data_load_metadata_uuid import get_last_data_load_metadata_uuid

postgresPort=db.postgresPort
postgresHost=db.postgresHost
postgresDB=db.postgresDB
postgresUser=db.postgresUser
postgresPassword=db.postgresPassword

segment_profile_poi_table_name=config.segment_profile_poi_table_name
poiDictCatTableName=config.poiDictCatTableName
individual_3p_table_name=config.individual_3p_table_name
segment_profile_app_table_name=config.segment_profile_app_table_name

# Database connection parameters
db_params = {
    'host': postgresHost,
    'port': postgresPort,
    'database': postgresDB,
    'user': postgresUser,
    'password': postgresPassword
}

connection_string=f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["database"]}'

# Function to find count of a specific segment_id
def count_segment(df, segment_id):
    count = df.reset_index()['segment_id'].value_counts()#df[df['segment_id'] == segment_id].shape[0]
    return count

def segment_profile_app_view_process():
    # df_apps_transform = pd.read_excel('3p_individual-16feb.xlsx',usecols=['cid','segment_id','no_of_apps'])
    df_apps_data =Helper.select_table_data(individual_3p_table_name,cols=[],connection_string=connection_string)
    df_apps_transform=df_apps_data.filter(['cid','segment_id','no_of_apps'], axis=1)

    # Get the UUID of the last data load metadata entry and ensure it's in the correct format
    data_load_metadata_uuid = get_last_data_load_metadata_uuid()

    # Initialize an empty list to store the rows for dataframe2
    df_app_view = []
    # Iterate over each row in dataframe1
    for _, row in df_apps_transform.iterrows():
        cid = row['cid']
        segment_id = row['segment_id']
        apps = ast.literal_eval(row['no_of_apps'])  # Convert the string representation to dictionary
    
    # Iterate over each category in apps
        for category, app_list in apps.items():
            # Iterate over each app in the app list
            for app_name in app_list:
                # Append the row to dataframe2_rows
                df_app_view.append({'cid': cid, 'segment_id': segment_id, 'category': category.split('_')[-1], 'app_name': app_name})

                # Create dataframe2 from the list of rows
    df_app_view = pd.DataFrame(df_app_view)
    seg_pop_per_attribute = count_segment(df_apps_transform, df_app_view['segment_id'])
    # Grouping and Aggregating to create dataframe2
    df_app_view_full = pd.DataFrame({
        'segment_id': df_app_view['segment_id'],
        'category': df_app_view['category'],
        'app_name': df_app_view['app_name'],
        'num_ind': df_app_view.groupby(['category', 'segment_id','app_name'])['category'].transform('count'),
        'num_ind_benchmark':df_app_view.groupby(['category','app_name'])['category'].transform('count'),
        'seg_pop': '',
        'tot_pop': df_apps_transform.shape[0], 
        'perc_ind':'',
        'index_cat':'',
        'data_load_metadata_id': [data_load_metadata_uuid] * len(df_app_view)  # Add the metadata UUID as a column   
                                })
    df_app_view_full['seg_pop']= df_app_view['segment_id'].map(seg_pop_per_attribute)
    df_app_view_full['perc_ind'] = df_app_view_full['num_ind']/df_app_view_full['seg_pop']
    temp_bench =  df_app_view_full['num_ind_benchmark']/df_app_view_full['tot_pop']
    df_app_view_full['index_cat'] = df_app_view_full['perc_ind']/temp_bench
    df_app_view_full = df_app_view_full.drop_duplicates()
    
    
    Helper.write_dataframe_to_postgresql_in_chunks(df_app_view_full,segment_profile_app_table_name, chunksize=10000, connection_string=connection_string)
