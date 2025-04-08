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

segment_geographics_table_name = config.segment_geographics_view_table_name
individual_3p_table_name=config.individual_3p_table_name
downloadedRawFilesLocation=config.locatPathToDownloadRawFiles
filename=config.demoggraphicsfilename

connection_string = f'postgresql://{postgresUser}:{postgresPassword}@{postgresHost}:{postgresPort}/{postgresDB}'
demographic_file=f'{downloadedRawFilesLocation}{filename}'


def segment_geographics_view_process():
    df = pd.read_csv(demographic_file)

    # Group by SegmentID and State and count the occurrences
    segment_state_counts = df.groupby(['SegmentID', 'state_name']).size().reset_index(name='state_count')

    # Group by SegmentID and calculate the total count for each SegmentID
    segment_total_counts = df.groupby('SegmentID').size().reset_index(name='total_count')

    # Merge the two dataframes on SegmentID
    merged_df = pd.merge(segment_state_counts, segment_total_counts, on='SegmentID')
    merged_df['country'] = 'USA'
    # Calculate the percentage of states mapped to each SegmentID(view1)
    merged_df['state_perc'] = (merged_df['state_count'] / merged_df['total_count']) * 100

    # Group by SegmentID, State, and county, and count the occurrences of counties
    county_state_counts = df.groupby(['SegmentID', 'state_name', 'county']).size().reset_index(name='county_count')

    # Merge the county counts with the merged dataframe
    merged_df = pd.merge(merged_df, county_state_counts, on=['SegmentID', 'state_name'])

    # Calculate the county_state_perc(view2)
    merged_df['county_state_perc'] = (merged_df['county_count'] / merged_df['state_count']) * 100

    # Calculate the county_country_perc(view2)
    county_country_counts = df.groupby(['state_name', 'county']).size().reset_index(name='county_total')
    merged_df = pd.merge(merged_df, county_country_counts, on=['state_name', 'county'])
    merged_df['county_country_perc'] = (merged_df['county_count'] / merged_df['total_count']) * 100

    # Group by SegmentID, county, and zip code, and count the occurrences of zip codes
    zip_county_counts = df.groupby(['SegmentID', 'county', 'Zip']).size().reset_index(name='zip_county_state_count')

    # Merge the zip counts with the merged dataframe
    merged_df = pd.merge(merged_df, zip_county_counts, on=['SegmentID', 'county'], how='left')

    # Calculate the zip_county_perc(view3)
    county_total_counts = df.groupby(['SegmentID', 'county']).size().reset_index(name='county_total_count')
    merged_df = pd.merge(merged_df, county_total_counts, on=['SegmentID', 'county'])
    merged_df['zip_county_perc'] = (merged_df['zip_county_state_count'] / merged_df['county_total_count']) * 100

    # Calculate Zip_state_perc (view3)
    merged_df['zip_state_perc'] = merged_df.apply(lambda row: (row['zip_county_state_count'] / row['state_count']) * 100 if pd.notnull(row['zip_county_state_count']) else 0, axis=1)

    # Calculate Zip_country_perc(view3)
    country_total_counts = df.groupby(['SegmentID']).size().reset_index(name='country_total_count')
    merged_df = pd.merge(merged_df, country_total_counts, on=['SegmentID'])
    merged_df['zip_country_perc'] = (merged_df['zip_county_state_count'] / merged_df['country_total_count']) * 100

    merged_df.rename(columns={'SegmentID': 'segment_id'}, inplace=True)
    merged_df.rename(columns={'Zip': 'zip_code'}, inplace=True)


    result = merged_df[['segment_id', 'country','state_name' ,'state_count', 'state_perc', 'county','county_count', 'county_state_perc', 'county_country_perc', 'zip_code', 'zip_county_state_count','zip_county_perc', 'zip_state_perc', 'zip_country_perc']]


    Helper.write_dataframe_to_postgresql_in_chunks(result, segment_geographics_table_name, chunksize=10000, connection_string=connection_string)



