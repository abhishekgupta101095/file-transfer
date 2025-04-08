# Copyright © 2024 Accenture. All rights reserved.

# File: re_dimension_sa360.py
# Author: abhishek.cw.gupta
# Date: 28-May-2024
# Description: Populate re_dimensions with SA360 data.

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
    list_met = ['adWordsConversionValue', 'adWordsConversions', 'adWordsViewThroughConversions', 'avgCpc', 'avgCpm', 'avgPos', 'clicks', 'cost', 'ctr', 'impr', 'visits']
    re_dimensions['id']= [178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223]
    re_dimensions['name']= ['effectiveBidStrategyId', 'deviceSegment', 'adId', 'adGroupId', 'accountId', 'advertiserId', 'agencyId', 'campaignId', 'keywordEngineId', 'date', 'adWordsConversionValue', 'adWordsConversions', 'adWordsViewThroughConversions', 'avgCpc', 'avgCpm', 'avgPos', 'clicks', 'cost', 'ctr', 'impr', 'keywordId', 'visits', 'dfaActions', 'dfaRevenue', 'dfaTransactions', 'dfaWeightedActions', 'floodlightActivityId', 'floodlightGroupId', 'bidStrategyInherited', 'creationTimestamp', 'effectiveKeywordMaxCpc', 'effectiveLabels', 'engineStatus', 'keywordLabels', 'keywordText', 'lastModifiedTimestamp', 'qualityScoreCurrent', 'status', 'searchImpressionShare', 'searchRankLostImpressionShare', 'qualityScoreAvg', 'accountType', 'creative', 'audience', 'campaignName', 'expectedImpression']
    re_dimensions['is_metric']= False
    re_dimensions.loc[re_dimensions['name'].isin(list_met),'is_metric']=True
    re_dimensions['is_dimension']= True
    re_dimensions.loc[re_dimensions['name'].isin(list_met),'is_dimension']=False
    re_dimensions.loc[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 44],'description']= ['Unique identifier of BidStrategyId', 'refers to a way to categorize and analyze campaign performance based on the type of device users interacted with your ads', 'Unique identifier of AdId', 'Unique identifier of AdGroupId', 'Unique identifier of AccountId', 'Unique identifier of AdvertiserId', 'Unique identifier of AgencyId', 'Unique identifier of CamapignId', 'unique identifier assigned to a keyword by the search engine itself (like Google Ads, Bing Ads etc.). ', 'captures the monetary value associated with a conversion.', 'The data that the advertiser has set up to be reported in the Google Ads Conversions column. When an Google Ads conversion action is created, the advertiser can choose whether to count those conversions in the Conversions reporting column.', 'The total number of Google Ads view-through conversions.', "The total cost of all clicks divided by the total number of clicks received. This metric is a monetary value and returned in the customer's currency by default.", "Average cost-per-thousand impressions (CPM). This metric is a monetary value and returned in the customer's currency by default.", "average position your ad occupies on a webpage or app where it's displayed in SA360 campaigns.A lower average position (closer to 1) generally signifies a more prominent ad placement.", 'counts the total number of times users clicked on your ad in SA360 campaigns.', "The sum of your cost-per-click (CPC) and cost-per-thousand impressions (CPM) costs during this period. This metric is a monetary value and returned in the customer's currency by default.", 'The number of clicks your ad receives (Clicks) divided by the number of times your ad is shown (Impressions)', 'Count of how often your ad has appeared on a search results page or website on the Google Network', 'unique identifier assigned to a keyword within a campaign.', "Clicks that Search Ads 360 has successfully recorded and forwarded to an advertiser's landing page", 'total number of conversions or actions tracked within DoubleClick for Advertisers(DFA) for your campaigns. ', 'total revenue generated from your campaigns as measured by DFA. ', 'total number of transactions that occurred as a result of campaigns according to DFA data.', 'number of weighted actions attributed to ad impressions. In DFA, we can assign weights to different conversion actions to reflect their relative importance to business.', 'unique identifier for a specific floodlight activity. Floodlight activities are essentially conversion tracking settings within DFA that define the actions you want to track on your website or app (e.g., purchases, sign-ups, video completions).', 'unique identifier for a floodlight group. Floodlight groups are a way to organize your floodlight activities into categories. For example, you might have a floodlight group for all purchase-related activities, another group for lead capture activities', 'This column indicates whether the keyword inherits its bid strategy from an ad group or campaign. A value of "TRUE" means the keyword inherits the bid strategy, while "FALSE" means it has an individually set bid strategy.', 'date and time when the keyword was first created in your Google Ads account.', 'This column represents the maximum amount you are willing to pay for a click on that specific keyword, considering any applicable bid adjustments.', 'lists any labels that are applied to the keyword, along with any labels inherited from the ad group or campaign. Labels are used to categorize and organize your keywords.', 'the current approval status of the keyword in the Google Ads engine. Possible values include "APPROVED", "DISAPPROVED", "PENDING_REVIEW", and others.', 'This column specifically lists the labels that are directly applied to the keyword itself, excluding any inherited labels.', 'contains the actual text of the keyword', 'date and time when the keyword was last edited in your Google Ads account.', "reflects Google's assessment of the quality and relevance of the keyword to your ad and landing page. A higher quality score can lead to lower costs and better ad positions.", 'shows the current operational status of the keyword. Possible values include "PAUSED", "ENABLED", and "REMOVED".', 'indicates the percentage of impressions your ad received for relevant searches compared to the total number of possible impressions.', 'impression share you lost due to your ad rank (not appearing high enough on the search results page).', 'average quality score of the keyword over a specific period (depending on your data export settings).', 'type of Google Ads account you are using, such as "SEARCH", "SHOPPING", or "VIDEO".', "The name of the advertising campaign you're analyzing."]
    re_dimensions['can_be_aggregated']= False
    re_dimensions['data_type']= ['STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'DATE', 'INTEGER', 'INTEGER', 'INTEGER', 'INTEGER', 'INTEGER', 'INTEGER', 'INTEGER', 'INTEGER', 'INTEGER', 'INTEGER', 'STRING', 'INTEGER', 'INTEGER', 'INTEGER', 'INTEGER', 'INTEGER', 'STRING', 'STRING', 'BOOLEAN', 'TIMESTAMP', 'INTEGER', 'STRING', 'STRING', 'STRING', 'STRING', 'TIMESTAMP', 'FLOAT', 'STRING', 'FLOAT', 'FLOAT', 'INTEGER', 'STRING', 'STRING', 'STRING', 'STRING', 'FLOAT']
    re_dimensions['is_geography_dependent']=False
    re_dimensions['vendor']= 'Google'    
    re_dimensions['category']= 'SA360'
    re_dimensions['domain_id'] = domains[domains['domain_name'].str.contains('Media')]['domain_id'][0]

    Helper.write_dataframe_to_postgresql_in_chunks(re_dimensions, re_dimensions_table_name, chunksize=10000, connection_string=connection_string)

if __name__ == '__main__':
   populate_re_dimensions()
