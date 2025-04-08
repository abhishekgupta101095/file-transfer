'''
Create segment view for each attribute of 3P demographic data
'''

import pandas as pd
import scripts.Helper as Helper

import scripts.dbConfig as db

import scripts.config as config
from scripts.get_data_load_metadata_uuid import get_last_data_load_metadata_uuid
import ast

postgresPort=db.postgresPort
postgresHost=db.postgresHost
postgresDB=db.postgresDB
postgresUser=db.postgresUser
postgresPassword=db.postgresPassword

segment_demo_insights_table_name=config.segment_demo_insights_table_name

# Database connection parameters
db_params = {
    'host': postgresHost,
    'port': postgresPort,
    'database': postgresDB,
    'user': postgresUser,
    'password': postgresPassword
}

connection_string=f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["database"]}'

def custom_list_parser(string):
    # Check if the string starts and ends with square brackets
    if (string.startswith('{') and string.endswith('}')) | (string.startswith('[') and string.endswith(']')):
        # Remove the brackets and split the string by commas
        elements = string[1:-1].split(',')
        # Strip whitespace from each element and return as a list
        return [element.strip() for element in elements]
    else:
        # Return the original string if it doesn't match the expected format
        return string


def process_data_view_for_attribute(attributename):
    '''
    Process the data and create a segment view for the given attribute.

    Args:
        attributename (str): The name of the attribute to create the segment view for.

    Returns:
        pandas.DataFrame: The segment view dataframe for the given attribute.
    '''
    df_3p_individual = Helper.select_table_data('3p_individual',cols=[],connection_string=connection_string)
    df = df_3p_individual.groupby([attributename, 'segment_id']).size().reset_index(name='count')

    # Get the UUID of the last data load metadata entry and ensure it's in the correct format
    data_load_metadata_uuid = get_last_data_load_metadata_uuid()
    
    if(attributename == 'affinity'):
        print(attributename)
        df_affinity = df_3p_individual 
        # Convert 'affinity' column to list
        #df_affinity['affinity'] = df_affinity['affinity'].apply(ast.literal_eval)
        #df_affinity['affinity'] = df_affinity['affinity'].tolist()
        df_affinity['affinity'] = df_affinity['affinity'].apply(custom_list_parser)
        # Explode 'affinity' column
        df_affinity = df_affinity.explode('affinity')
        # Group by 'segment_id', 'affinity', and count records
        df_affinities = df_affinity.groupby(['segment_id', 'affinity']).size().reset_index(name='record_count')        
        # Create 'sortcolumn' and 'sortdata' columns
        df_affinities['sortcolumn'] = 'affinity'
        df_affinities['sortdata'] = df_affinities['affinity']
        df_affinities['sortdata'] = df_affinities['sortdata'].str.strip("'")

        # Sorting the DataFrame
        df_affinities.sort_values(by=['sortcolumn', 'segment_id', 'sortdata'], inplace=True)
                 
        df_affinities['rounded_percentage'] = ''
        df_affinities['rank'] = ''
       
        seg_pop_per_attribute = df.groupby('segment_id')['count'].sum()
        df_affinities['seg_pop'] =  df['segment_id'].map(seg_pop_per_attribute)
        df_affinities['tot_pop'] = df['count'].sum() 

        # Calculate the benchmark
        benchmark = df_affinities.groupby('sortdata')['record_count'].sum()

        # Map the benchmark values to the original dataframe
        df_affinities['benchmark'] = df_affinities['sortdata'].map(benchmark)
        percentage = (df_affinities['record_count'] / df_affinities['seg_pop'] ) 
        percentage_benchmark = (df_affinities['benchmark'] / df_affinities['tot_pop'] )
        df_affinities['rounded_percentage']  = percentage/percentage_benchmark       
        df_affinities['rank'] = df_affinities.groupby('segment_id')['rounded_percentage'].rank(ascending=False).astype(int)
        df_affinities = df_affinities.groupby('segment_id').apply(lambda x: x.sort_values(by='rank', ascending=True)).reset_index(drop=True)
        df_affinities['flag'] = 'I' # index value
        df_affinities.drop(['affinity'], axis=1, inplace=True)
        df_affinities.drop(['benchmark'], axis=1, inplace=True) 
        df_affinities['data_load_metadata_id'] = [data_load_metadata_uuid] * len(df_affinities)  # Add the metadata UUID as a column
        df_affinities = df_affinities[['sortcolumn','sortdata','segment_id', 'record_count','rounded_percentage','seg_pop','tot_pop','rank','flag','data_load_metadata_id']]
        return df_affinities      

    else:
        # df_3p_individual = pd.read_excel('3p_individual.xlsx')
        # df = df_3p_individual.groupby([attributename, 'segment_id']).size().reset_index(name='count')
        df['percentage'] = (df['count'] / df.groupby('segment_id')['count'].transform('sum')) * 100
        df.columns = ['sortcolumn', 'sordata', 'count', 'percentage']

        df['percentage'] = df['percentage'].round(2)  # Calculate rank based on rounded percentage
        df.rename(columns={'sordata': 'prediction'}, inplace=True)
        df.rename(columns={'sortcolumn': 'sortdata'}, inplace=True)
        df.insert(0, 'sortcolumn', attributename)
        seg_pop_per_attribute = df.groupby('prediction')['count'].sum()
        df['seg_pop'] =  df['prediction'].map(seg_pop_per_attribute)
        df['tot_pop'] = df['count'].sum()  
        df['sortcolumn'] = attributename
        df['rank'] = df.groupby('prediction')['count'].rank(ascending=False).astype(int)
        df = df.groupby('prediction').apply(lambda x: x.sort_values(by='rank', ascending=True)).reset_index(drop=True)
        df['flag'] = 'P' #percentage value
        df.rename(columns={'prediction': 'segment_id'}, inplace=True)
        df['data_load_metadata_id'] = [data_load_metadata_uuid] * len(df)  # Add the metadata UUID as a column 
        return df
    print(df)

# Read 3p_individual data in dataframe 					
#df_3p_individual = pd.read_excel('3p_individual.xlsx', sheet_name='Sheet1', usecols=['cid', 'segment_id', 'age_group', 'education_level', 'gender', 'home_owner_status', 'household_income', 'individual_income', 'net_worth', 'married', 'no_of_adults_hh', 'no_of_children_hh', 'premium_card_holder', 'collar_worker', 'segment_id', 'urbanicity', 'occupation'])
#print(df_3p_individual.columns)

def segment_demo_insights_view_process():
    df_married = process_data_view_for_attribute('married')
    df_no_of_adults_hh = process_data_view_for_attribute('no_of_adults_hh')
    df_no_of_children_hh = process_data_view_for_attribute('no_of_children_hh')
    df_premium_card_holder = process_data_view_for_attribute('premium_card_holder')
    df_collar_worker = process_data_view_for_attribute('collar_worker')
    df_urbanicity = process_data_view_for_attribute('urbanicity')
    df_occupation = process_data_view_for_attribute('occupation')
    df_age_group = process_data_view_for_attribute('age_group')
    df_education_level = process_data_view_for_attribute('education_level')
    df_gender = process_data_view_for_attribute('gender')
    df_home_owner_status = process_data_view_for_attribute('home_owner_status')
    df_household_income = process_data_view_for_attribute('household_income')
    df_individual_income = process_data_view_for_attribute('individual_income')
    df_individual_net_worth = process_data_view_for_attribute('net_worth')
    df_individual_affinity = process_data_view_for_attribute('affinity')
 
    result = pd.concat([df_married, df_no_of_adults_hh, df_no_of_children_hh, df_premium_card_holder, df_urbanicity, df_collar_worker, df_occupation, df_age_group, df_education_level, df_gender, df_home_owner_status, df_household_income, df_individual_income, df_individual_net_worth], ignore_index=True)
    result.rename(columns={'count': 'record_count', 'percentage': 'rounded_percentage'}, inplace=True)
    result = pd.concat([result, df_individual_affinity], ignore_index=True)
    print(result)
    #df_result.to_csv('df_result.csv', index=False)    
    Helper.write_dataframe_to_postgresql_in_chunks(result,segment_demo_insights_table_name, chunksize=10000, connection_string=connection_string)
