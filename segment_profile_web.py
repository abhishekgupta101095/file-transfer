'''
Create segment view for each attribute of 3P demographic data
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

segment_profile_web_table_name=config.segment_profile_web_table_name

individual_3p_table_name = config.individual_3p_table_name
webDictCatTableName=config.webDictCatTableName

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

def segment_profile_web_view_process():
    df_web_category =Helper.select_table_data(webDictCatTableName,cols=[],connection_string=connection_string)
    df_web_category['category_Tier1'] = df_web_category['category'].str.cat(df_web_category['tier1'], sep='_')
    df_web_category['category_Tier1_Tier2'] = df_web_category['category_Tier1'].str.cat(df_web_category['tier2'], sep='_')
    
    df_3p_original =Helper.select_table_data(individual_3p_table_name,cols=[],connection_string=connection_string)
    df_individual = df_3p_original.filter(['cid','segment_id','num_domain_surf'], axis=1)    
    df_individual['num_domain_surf'] = [ast.literal_eval(num_domain_surf_str.replace('Num_DomainSurf_', '')) for num_domain_surf_str in df_individual['num_domain_surf']]
    # Convert the dictionary into a list of dictionaries
    list_of_dicts = []
    for cid,segment_id, num_domain_surf_dict in zip(df_individual['cid'],df_individual['segment_id'], df_individual['num_domain_surf']):
        for key, value in num_domain_surf_dict.items():
            list_of_dicts.append({'cid': cid,'segment_id':segment_id, 'tier2': key})
    
    # Create DataFrame from the list of dictionaries
    df_3p_individual = pd.DataFrame(list_of_dicts)
    df_3p_individual.rename(columns = {'tier2':'matching_data'}, inplace = True)
    
        
    new_tier2 = df_web_category[['category','tier1','tier2']]
    new_category_Tier1 = df_web_category[['category','tier1','tier2','category_Tier1']]
    new_category = df_web_category[['category','tier1','tier2']]
    new_category_Tier1_Tier2 = df_web_category[['category','tier1','tier2','category_Tier1','category_Tier1_Tier2']]
    
    
    merged_df = pd.merge(df_3p_individual, new_tier2, left_on='matching_data', right_on='tier2', how='left',indicator= True)
    merged_df.dropna(subset=['tier2'], inplace=True)
    
    merged1_df = pd.merge(df_3p_individual, new_category_Tier1, left_on='matching_data', right_on='category_Tier1', how='left' , indicator= True)
    merged1_df.dropna(subset=['category_Tier1'], inplace=True)
    
    merged2_df = pd.merge(df_3p_individual, new_category, left_on='matching_data', right_on='category', how='left',indicator= True)
    merged2_df.dropna(subset=['category'], inplace=True)
    
    merged3_df = pd.merge(df_3p_individual, new_category_Tier1_Tier2, left_on='matching_data', right_on='category_Tier1_Tier2', how='left',indicator= True)
    merged3_df.dropna(subset=['category_Tier1_Tier2'], inplace=True)
    
    result = pd.concat([merged_df, merged1_df, merged2_df,merged3_df], ignore_index=True)
    result = result.drop_duplicates()
    # Specify columns to remove
    columns_to_remove = ['cid','_merge','category_Tier1','category_Tier1_Tier2','matching_data']
    
    # Remove columns using `drop`
    result = result.drop(columns_to_remove, axis=1)  # axis=1 for columns
    result = result.drop_duplicates()

    seg_pop_per_attribute = count_segment(df_individual, result['segment_id'])
    # Get the UUID of the last data load metadata entry and ensure it's in the correct format
    data_load_metadata_uuid = get_last_data_load_metadata_uuid()
    # Grouping and Aggregating to create dataframe2
    full = pd.DataFrame({
        'segment_id': result['segment_id'],
        'category': result['category'],
        'tier1': result['tier1'],
        'tier2': result['tier2'],    
        'num_ind_cat': result.groupby(['category', 'segment_id'])['category'].transform('count'),
        'num_ind_tier1': result.groupby(['category', 'tier1','segment_id'])['category'].transform('count'),
        'num_ind_tier2': result.groupby(['category', 'tier1', 'tier2','segment_id'])['category'].transform('count'),
    
        'num_ind_tier2_benchmark':result.groupby(['category','tier1','tier2'])['category'].transform('count'),
        'num_ind_tier1_benchmark':result.groupby(['category','tier1'])['tier1'].transform('count'),
        'num_ind_cat_benchmark':result.groupby(['category'])['category'].transform('count'),
    
        'seg_pop': '',
        'tot_pop': df_3p_original.shape[0],
        'perc_ind_cat':"",
        'perc_ind_tier1':"",
        'perc_ind_tier2':"",
        'index_tier2': "",
        'index_tier1':"",
        'index_cat': "",
        'data_load_metadata_id': [data_load_metadata_uuid] * len(result)  # Add the metadata UUID as a column   
    
    })
    full = full.drop_duplicates()
    full['seg_pop']= result['segment_id'].map(seg_pop_per_attribute)
    full['perc_ind_cat'] = full['num_ind_cat'] / full['seg_pop']
    full['perc_ind_tier1'] = full['num_ind_tier1'] / full['seg_pop']
    full['perc_ind_tier2'] = full['num_ind_tier2'] / full['seg_pop']
    
    temp_bench_tier2 = full['num_ind_tier2_benchmark'] /full['tot_pop']
    temp_bench_tier1 = full['num_ind_tier1_benchmark'] /full['tot_pop']
    temp_bench_cat = full['num_ind_cat_benchmark'] /full['tot_pop']
    
    full['index_tier2'] = full['perc_ind_tier2'] / temp_bench_tier2  
    full['index_tier1'] = full['perc_ind_tier1'] / temp_bench_tier1
    full['index_cat'] =  full['perc_ind_cat'] / temp_bench_cat
    temp_bench = ""
    temp_bench_cat =""
    temp_bench_tier1 =""
    
    
    columns_to_remove1 = ['num_ind_tier2_benchmark','num_ind_tier1_benchmark','num_ind_cat_benchmark']
    result = full.drop(columns_to_remove1, axis=1)  # axis=1 for columns
            
      
    Helper.write_dataframe_to_postgresql_in_chunks(result,segment_profile_web_table_name, chunksize=10000, connection_string=connection_string)


