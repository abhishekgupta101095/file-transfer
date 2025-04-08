# Copyright © 2024 Accenture. All rights reserved.

# File: re_dimension_cm360.py
# Author: abhishek.cw.gupta
# Date: 28-May-2024
# Description: Populate re_dimensions with CM360 data.

# import pandas as pd
# import numpy as np
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

re_dimensions_table_name = config.re_dimensions_table_name
domains_table_name = config.domains_table_name

connection_string = f'postgresql://{postgresUser}:{postgresPassword}@{postgresHost}:{postgresPort}/{postgresDB}'

def populate_re_dimensions():

    # Fetch the dataset
    domains = Helper.select_table_data(domains_table_name, cols=[], connection_string=connection_string)
    re_dimensions = pd.DataFrame()
    list_met = ['total_revenue']
    re_dimensions['id']= [115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177]
    re_dimensions['name']= ['advertiser_id', 'campaign_id', 'ad_id', 'user_id', 'advertiser', 'advertiser_group_id', 'advertiser_group', 'campaign', 'campaign_start_date', 'campaign_end_date', 'ad', 'click_through_url', 'ad_type', 'keyword', 'creative_version', 'placement_id', 'package_roadblock_total_booked_units', 'placement_rate', 'placement_cost_structure', 'browser_id', 'browser_platform', 'browser_version', 'operating_system_id', 'operating_system', 'dma_id', 'site_id', 'city_id', 'city', 'state', 'country_code', 'zip_postal_code', 'asset_id', 'rendering_id', 'creative_id', 'creative', 'impression_id', 'active_view_eligible_impressions', 'active_view_measurable_impressions', 'active_view_viewable_impressions', 'impression_event_type', 'impression_event_sub_type', 'impression_event_time', 'referrer_url', 'click_id', 'landing_page_url_id', 'click_event_type', 'click_event_sub_type', 'click_event_time', 'segment_value', 'activity_id', 'activity_event_type', 'activity_event_sub_type', 'activity_event_time', 'floodlight_configuration', 'activity_group', 'activity_type_id', 'activity_type', 'interaction_time', 'conversion_id', 'total_conversions', 'total_revenue', 'report_date', 'ingestiontime']
    re_dimensions['is_metric']= False
    re_dimensions.loc[re_dimensions['name'].isin(list_met),'is_metric']=True
    re_dimensions['is_dimension']= True
    re_dimensions.loc[re_dimensions['name'].isin(list_met),'is_dimension']=False
    re_dimensions['description'] = ['unique id of the advertiser', 'unique id of the campaign', 'unique id of the ad placement', 'the tracking id', 'user entered name of advertiser', 'group id to which this advertiser belong, an advertiser can be a member of only one group', 'user specified group to which this advertiser belong', 'user entered campaign name', 'campaign start date as yyyymmdd', 'campaign end date as yyyymmdd', 'user defined ad name', 'click through url of click ad type creatives only (aka click commands)', 'type of ad', 'targeted keyword', 'creative version number', 'unique id for the site page / placement where the ad ran', 'purchased units', "purchased rate expressed in nanos of the network's currency", 'cost structure of placement', 'id of the browser type', 'browser name', 'version of the browser', 'id of the operating system', 'operating system name', 'designated market area id', 'unique id for the site where the ad ran', 'city id', 'city name', "id for user's state or province", '2-letter iso 3166-1 country code', 'postal-code', 'the unique id of the asset', 'unique id of the creative for data transfer.', 'matches the creative id in dcm ui', 'creative name', 'unique id used to join impression, click, and rich media, data transfer file data', 'whether the impression was eligible to measure viewability', 'whether the impression was measurable with active view', 'whether the impression was viewable', 'contains details related to the event', 'contains details related to the event', 'time in microseconds', 'contains the url on which the ad appeared', 'a tag passed with ad clicks, to identify the campaign associated with the ad for ad tracking', 'unique id of the landing page url', 'contains details related to the event', 'contains details related to the event', 'time in microseconds', 'search ads 360 keyword id', 'the id of the floodlight tag related to the conversion event', 'contains details related to the event', 'contains details related to the event', 'time in microseconds', 'unique id of the floodlight configuration', 'name for this activity group', 'id for this activity group', 'user defined string corresponding to type', 'the activity’s associated click or impression time in microseconds', 'indicates whether an activity has been matched as a post-click or post-impression or unmatched', 'contains additional information on conversions', 'contains additional information on revenue', 'date of the recieved data', 'last update date']
    re_dimensions['can_be_aggregated']= False
    re_dimensions.loc[re_dimensions['name']=='total_revenue','can_be_aggregated']= True
    re_dimensions['data_type']= ['INTEGER', 'INTEGER', 'INTEGER', 'STRING', 'STRING', 'INTEGER', 'STRING', 'STRING', 'DATE', 'DATE', 'STRING', 'STRING', 'STRING', 'STRING', 'INTEGER', 'INTEGER', 'INTEGER', 'FLOAT', 'STRING', 'INTEGER', 'STRING', 'STRING', 'INTEGER', 'STRING', 'INTEGER', 'INTEGER', 'INTEGER', 'STRING', 'STRING', 'STRING', 'STRING', 'INTEGER', 'INTEGER', 'INTEGER', 'STRING', 'STRING', 'INTEGER', 'INTEGER', 'INTEGER', 'STRING', 'STRING', 'INTEGER', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'INTEGER', 'INTEGER', 'INTEGER', 'STRING', 'STRING', 'INTEGER', 'INTEGER', 'STRING', 'INTEGER', 'STRING', 'STRING', 'INTEGER', 'INTEGER', 'FLOAT', 'TIMESTAMP', 'TIMESTAMP']
    re_dimensions['is_geography_dependent']=False
    re_dimensions.loc[[26,27,28,29,30],'is_geography_dependent']= True
    re_dimensions['vendor']= 'Google'    
    re_dimensions['category']= 'CM360'
    re_dimensions['domain_id'] = domains[domains['domain_name'].str.contains('Media')]['domain_id'][0]

    Helper.write_dataframe_to_postgresql_in_chunks(re_dimensions, re_dimensions_table_name, chunksize=10000, connection_string=connection_string)

if __name__ == '__main__':
   populate_re_dimensions()
