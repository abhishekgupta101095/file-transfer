'''
Create segment view for segment_profile_social
'''

import pandas as pd
import ast
from ast import literal_eval
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
segment_profile_social_table_name=config.segment_profile_social_table_name
social_meta_table_name=config.social_meta_table_name

# Database connection parameters
db_params = {
    'host': postgresHost,
    'port': postgresPort,
    'database': postgresDB,
    'user': postgresUser,
    'password': postgresPassword
}

connection_string=f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["database"]}'


# Function to convert list items to rows
def explode_social(row):
    social_list = row['social']
    #print("Social list is:")
    #print(social_list)
    if isinstance(social_list, str):
        #social_list = social_list.replace('nan,', "'brands_travel'")
        #print("social list is a list")
        #print(social_list)
        social_list = literal_eval(social_list)
    elif not isinstance(social_list, list):
        #print("social list is not  a list")
        social_list = [social_list]  # Convert non-list values to list
    return [(row['cid'], row['segment_id'], social) for social in social_list]

# Function to find count of a specific segment_id
def count_segment(df, segment_id):
    count = df.reset_index()['segment_id'].value_counts()#df[df['segment_id'] == segment_id].shape[0]
    return count

def segment_profile_social_view_process():
    # df_apps_transform = pd.read_excel('3p_individual-16feb.xlsx',usecols=['cid','segment_id','no_of_apps'])
    df_social_category =Helper.select_table_data(social_meta_table_name,cols=[],connection_string=connection_string)
    df_social_category['cat1_cat2'] = df_social_category['category1'].str.cat(df_social_category['category2'], sep='_')
    df_social_category['cat1_cat2_cat3'] = df_social_category['cat1_cat2'].str.cat(df_social_category['category3'], sep='_')
    df_social_category['cat1_cat2_Cat3_cat6'] = df_social_category['cat1_cat2_cat3'].str.cat(df_social_category['category6'], sep='_')
    
    df_original =Helper.select_table_data(individual_3p_table_name,cols=[],connection_string=connection_string)
    df_individual = df_original[['cid','segment_id','social']]
    print(df_individual)
    # Apply the function to each row and flatten the list of tuples
    rows = df_individual.apply(explode_social, axis=1).explode()

    # Create the new DataFrame
    df_3p_individual = pd.DataFrame(rows.tolist(), columns=['cid', 'segment_id', 'matching_data']) 
    new_cat1_cat2 = df_social_category[['category1','category2','category3','category6','cat1_cat2']]
    new_cat1_cat2_Cat3_cat6 = df_social_category[['category1','category2','category3','category6','cat1_cat2_Cat3_cat6']]
    new_cat6= df_social_category[['category1','category2','category3','category6']]

    
    merged_df = pd.merge(df_3p_individual, new_cat1_cat2_Cat3_cat6, left_on='matching_data', right_on='cat1_cat2_Cat3_cat6', how='left',indicator= True)
    merged_df = merged_df.drop_duplicates()

    # Filter dataframe2 to only include rows where Category 3 is blank
    filtered_new_cat1_cat2 = new_cat1_cat2[new_cat1_cat2['category3'].isnull()] 
    filtered_new_cat1_cat2.to_csv('filtered_new_cat1_cat2.csv', index=False)  # Optional: Save to a CSV file
    merged2_df = pd.merge(df_3p_individual, filtered_new_cat1_cat2, left_on='matching_data', right_on='cat1_cat2', how='left',indicator= True)
    merged2_df.dropna(subset=['cat1_cat2'], inplace=True)
    merged2_df = merged2_df.drop_duplicates()

    result = pd.concat([merged_df, merged2_df], ignore_index=True)
    filtered_result= result[result['category1'].notna()] 
    result = result.drop_duplicates()


    # # Specify columns to remove
    columns_to_remove = ['cid','_merge','cat1_cat2_Cat3_cat6','cat1_cat2','matching_data']

    # Remove columns using `drop`
    filtered_result = filtered_result.drop(columns_to_remove, axis=1)  # axis=1 for columns
    filtered_result = filtered_result.drop_duplicates()
    seg_pop_per_attribute = count_segment(df_individual, filtered_result['segment_id'])
    # Get the UUID of the last data load metadata entry and ensure it's in the correct format
    data_load_metadata_uuid = get_last_data_load_metadata_uuid()
    # Grouping and Aggregating to create result dataframe
    full = pd.DataFrame({
        'segment_id': filtered_result['segment_id'],
        'cat1': filtered_result['category1'],
        'cat2': filtered_result['category2'],
        'cat3': filtered_result['category3'], 
        'cat6': filtered_result['category6'],    
        'num_ind_cat1': filtered_result.groupby(['category1', 'segment_id'])['category1'].transform('count'),
        'num_ind_cat2': filtered_result.groupby(['category1', 'category2','segment_id'])['category1'].transform('count'),
        'num_ind_cat3': filtered_result.groupby(['category1', 'category2', 'category3','segment_id'])['category1'].transform('count'),
        'num_ind_cat6': filtered_result.groupby(['category1', 'category2', 'category3','category6','segment_id'])['category1'].transform('count'),

        'num_ind_cat1_benchmark':filtered_result.groupby(['category1'])['category1'].transform('count'),
        'num_ind_cat2_benchmark':filtered_result.groupby(['category1', 'category2'])['category1'].transform('count'),
        'num_ind_cat3_benchmark':filtered_result.groupby(['category1', 'category2', 'category3'])['category1'].transform('count'),
        'num_ind_cat6_benchmark':filtered_result.groupby(['category1', 'category2', 'category3','category6'])['category1'].transform('count'),

        'seg_pop': '',#filtered_result.groupby(['segment_id'])['category1'].transform('count'),
        'tot_pop': df_original.shape[0], #filtered_result.shape[0],
        'perc_ind_cat1':"",
        'perc_ind_cat2':"",
        'perc_ind_cat3':"",
        'perc_ind_cat6':"",
        'index_cat1': "",
        'index_cat2':"",
        'index_cat3': "",
        'index_cat6': "",
        'data_load_metadata_id': [data_load_metadata_uuid] * len(filtered_result)  # Add the metadata UUID as a column    

    })
    full['seg_pop']= filtered_result['segment_id'].map(seg_pop_per_attribute)
    full.to_csv('full.csv', index=False)  # Optional: Save to a CSV file
    # full = full.drop_duplicates()
    full['perc_ind_cat1'] = full['num_ind_cat1'] / full['seg_pop']
    full['perc_ind_cat2'] = full['num_ind_cat2'] / full['seg_pop']
    full['perc_ind_cat3'] = full['num_ind_cat3'] / full['seg_pop']
    full['perc_ind_cat6'] = full['num_ind_cat6'] / full['seg_pop']

    temp_bench_cat1 = full['num_ind_cat1_benchmark'] /full['tot_pop']
    temp_bench_cat2 = full['num_ind_cat2_benchmark'] /full['tot_pop']
    temp_bench_cat3 = full['num_ind_cat3_benchmark'] /full['tot_pop']
    temp_bench_cat6 = full['num_ind_cat6_benchmark'] /full['tot_pop']

    full['index_cat1'] = full['perc_ind_cat1'] / temp_bench_cat1  
    full['index_cat2'] = full['perc_ind_cat2'] / temp_bench_cat2
    full['index_cat3'] = full['perc_ind_cat3'] / temp_bench_cat3
    full['index_cat6'] = full['perc_ind_cat6'] / temp_bench_cat6
    temp_bench_cat1 = ""
    temp_bench_cat2 =""
    temp_bench_cat3 =""
    temp_bench_cat6 =""

    # full.to_csv('full.csv', index=False)  # Optional: Save to a CSV file

    
    Helper.write_dataframe_to_postgresql_in_chunks(full,segment_profile_social_table_name, chunksize=10000, connection_string=connection_string)
