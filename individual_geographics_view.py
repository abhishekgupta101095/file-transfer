import pandas as pd
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

individual_geographics_table_name = config.individual_geographics_view_table_name
individual_3p_table_name=config.individual_3p_table_name
downloadedRawFilesLocation=config.locatPathToDownloadRawFiles
filename=config.demoggraphicsfilename

connection_string = f'postgresql://{postgresUser}:{postgresPassword}@{postgresHost}:{postgresPort}/{postgresDB}'
demographic_file=f'{downloadedRawFilesLocation}{filename}'


def individual_geographics_view_process():
    df = pd.read_csv(demographic_file)

        #calcullating "state_perc"
    state_counts = df.groupby('state_name').size().reset_index(name='state_count')
    total_count = df.shape[0]
    state_counts['total_count'] = total_count
    state_counts['state_perc'] = (state_counts['state_count'] / state_counts['total_count']) * 100


    county_counts = df.groupby(['state_name', 'county']).size().reset_index(name='county_count')

    #Calculating "county_state_perc"
    merged_df = pd.merge(state_counts, county_counts, on='state_name')


    merged_df['county_state_perc'] = (merged_df['county_count'] / merged_df['state_count']) * 100

    # Calculating "county_country_perc"
    county_total_counts = df.groupby('county').size().reset_index(name='county_total_count')
    county_total_counts['county_country_perc'] = (county_total_counts['county_total_count'] / total_count) * 100
    merged_df = pd.merge(merged_df, county_total_counts, on='county', how='left')

    zip_county_counts = df.groupby(['county', 'Zip']).size().reset_index(name='zip_county_count')


    zip_state_counts = df.groupby(['state_name', 'Zip']).size().reset_index(name='zip_state_count')


    zip_total_counts = df.groupby('Zip').size().reset_index(name='zip_total_count')


    merged_df = pd.merge(merged_df, zip_county_counts, on='county', how='left')
    merged_df = pd.merge(merged_df, zip_state_counts, on=['state_name', 'Zip'], how='left')
    merged_df = pd.merge(merged_df, zip_total_counts, on='Zip', how='left')

    # Calculating "zip_county_perc"
    merged_df['zip_county_perc'] = (merged_df['zip_county_count'] / merged_df['county_count']) * 100

    # Calculating "zip_state_perc"
    merged_df['zip_state_perc'] = (merged_df['zip_state_count'] / merged_df['state_count']) * 100

    # Calculating "zip_country_perc"
    merged_df['zip_country_perc'] = (merged_df['zip_total_count'] / total_count) * 100


    merged_df['country'] = 'USA'


    result = merged_df[['country', 'state_name', 'state_count', 'state_perc', 'county', 'county_count', 
                        'county_state_perc', 'county_country_perc', 'Zip', 'zip_county_count', 'zip_county_perc',
                        'zip_state_count','zip_state_perc', 'zip_country_perc']]


    result = result.dropna(subset=['zip_state_count'])


    result.rename(columns={'Zip': 'zip_code'}, inplace=True)


    Helper.write_dataframe_to_postgresql_in_chunks(result, individual_geographics_table_name, chunksize=10000, connection_string=connection_string)



