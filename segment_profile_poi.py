'''
Create segment view for segment_profile_poi
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

def segment_profile_poi_view_process():
    df_poi_category =Helper.select_table_data(poiDictCatTableName,cols=[],connection_string=connection_string)
    
    df_3p_original =Helper.select_table_data(individual_3p_table_name,cols=[],connection_string=connection_string)
    df_individual = df_3p_original.filter(['cid','segment_id','visits'], axis=1)    
    df_individual['visits'] = [ast.literal_eval(visits_str.replace('# Visits to ', '')) for visits_str in df_individual['visits']]
    # Convert the dictionary into a list of dictionaries
    list_of_dicts = []
    for cid,segment_id, visits_dict in zip(df_individual['cid'],df_individual['segment_id'], df_individual['visits']):
        for key, value in visits_dict.items():
            list_of_dicts.append({'cid': cid,'segment_id':segment_id, 'visits': key})
     
    # Create DataFrame from the list of dictionaries
    df_3p_individual_poi = pd.DataFrame(list_of_dicts)
    df_3p_individual_poi.rename(columns = {'visits':'matching_data'}, inplace = True)
    
    new_location = df_poi_category[['category','subcategory','location']]
    
    merged_df = pd.merge(df_3p_individual_poi, new_location, left_on='matching_data', right_on='location', how='left',indicator= True)
    merged_df.dropna(subset=['location'], inplace=True)
    
    columns_to_remove1 = ['cid','matching_data','_merge']
    
    # Remove columns using `drop`
    result = merged_df.drop(columns_to_remove1, axis=1)  # axis=1 for columns
    result = result.drop_duplicates()
    seg_pop_per_attribute = count_segment(df_individual, result['segment_id'])

    # result.to_csv('merge_location.csv', index=False)  # Optional: Save to a CSV file
    
    # Get the UUID of the last data load metadata entry and ensure it's in the correct format
    data_load_metadata_uuid = get_last_data_load_metadata_uuid()

    # Grouping and Aggregating to create dataframe2
    full = pd.DataFrame({
        'segment_id': result['segment_id'],
        'top_category': result['category'],
        'sub_category': result['subcategory'],
        'location_name': result['location'],    
        'num_ind_loc': result.groupby(['category','location', 'segment_id'])['location'].transform('count'),
        'num_ind_subcat': result.groupby(['category','subcategory','segment_id'])['subcategory'].transform('count'),
        'num_ind_cat': result.groupby(['category','segment_id'])['category'].transform('count'),   
         
        'num_ind_loc_bench':result.groupby(['location'])['location'].transform('count'),  
        'num_ind_subcat_bench':result.groupby(['category','subcategory'])['subcategory'].transform('count'),     
        'num_ind_cat_bench':result.groupby(['category'])['category'].transform('count'), 
    
        'seg_pop': '',#result.groupby(['segment_id'])['category'].transform('count'),
        'tot_pop': df_3p_original.shape[0],
    
        'perc_ind_cat':"",
        'perc_ind_subcat':"",
        'perc_ind_loc':"",
        'index_loc': "",
        'index_subcat':"",
        'index_cat': "",
        'data_load_metadata_id': [data_load_metadata_uuid] * len(result)  # Add the metadata UUID as a column 
      
    
    })
    full = full.drop_duplicates()
    full['seg_pop']= result['segment_id'].map(seg_pop_per_attribute)
    # dataframe2['cat_percentage'] = (dataframe2['cat_count'] / total_rows) * 100
    full['perc_ind_cat'] = full['num_ind_cat'] / full['seg_pop']
    full['perc_ind_subcat'] = full['num_ind_subcat'] / full['seg_pop']
    full['perc_ind_loc'] = full['num_ind_loc'] / full['seg_pop']
    
    temp_bench_cat = full['num_ind_cat_bench'] /full['tot_pop']
    temp_bench_subcat = full['num_ind_subcat_bench'] /full['tot_pop']
    temp_bench_loc = full['num_ind_loc_bench'] /full['tot_pop']
    
    full['index_loc'] = full['perc_ind_loc'] / temp_bench_loc  
    full['index_subcat'] = full['perc_ind_subcat'] / temp_bench_subcat
    full['index_cat'] =  full['perc_ind_cat'] / temp_bench_cat
    
    temp_bench = ""
    temp_bench_cat =""
    temp_bench_tier1 =""
    
    
    columns_to_remove1 = ['num_ind_cat_bench','num_ind_subcat_bench','num_ind_loc_bench']
    
    # Remove columns using `drop`
    result = full.drop(columns_to_remove1, axis=1)  # axis=1 for columns
    
    Helper.write_dataframe_to_postgresql_in_chunks(result,segment_profile_poi_table_name, chunksize=10000, connection_string=connection_string)

