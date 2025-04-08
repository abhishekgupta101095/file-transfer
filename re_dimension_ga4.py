# Copyright © 2024 Accenture. All rights reserved.

# File: re_dimension_ga4.py
# Author: abhishek.cw.gupta
# Date: 28-May-2024
# Description: Populate re_dimensions with GA4 data.

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
    list_met = ['ecommerce_purchase_revenue','ecommerce_refund_value','ecommerce_shipping_value','ecommerce_tax_value','items_item_refund','items_item_revenue']
    re_dimensions['id']= [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,114]
    re_dimensions['name']= ['app_info_firebase_app_id', 'app_info_id', 'app_info_install_source', 'app_info_install_store', 'app_info_version', 'device_advertising_id', 'device_category', 'device_is_limited_ad_tracking', 'device_language', 'device_mobile_brand_name', 'device_mobile_marketing_name', 'device_mobile_model_name', 'device_mobile_os_hardware_model', 'device_operating_system', 'device_operating_system_version', 'device_time_zone_offset_seconds', 'device_vendor_id', 'device_web_info', 'device_web_info_browser', 'device_web_info_browser_version', 'ecommerce_purchase_revenue', 'ecommerce_purchase_revenue_in_usd', 'ecommerce_refund_value', 'ecommerce_refund_value_in_usd', 'ecommerce_shipping_value', 'ecommerce_shipping_value_in_usd', 'ecommerce_tax_value', 'ecommerce_tax_value_in_usd', 'ecommerce_total_item_quantity', 'ecommerce_transaction_id', 'ecommerce_unique_items', 'event_bundle_sequence_id', 'event_date', 'event_dimensions', 'event_dimensions_hostname', 'event_name', 'event_params_key', 'event_params_value', 'event_params_value_double_value', 'event_params_value_float_value', 'event_params_value_int_value', 'event_params_value_string_value', 'event_previous_timestamp', 'event_server_timestamp_offset', 'event_timestamp', 'event_value_in_usd', 'geo_city', 'geo_continent', 'geo_country', 'geo_metro', 'geo_region', 'geo_sub_continent', 'items_affiliation', 'items_coupon', 'items_creative_name', 'items_creative_slot', 'items_item_brand', 'items_item_category', 'items_item_category2', 'items_item_category3', 'items_item_category4', 'items_item_category5', 'items_item_id', 'items_item_list_id', 'items_item_list_index', 'items_item_list_name', 'items_item_name', 'items_item_refund', 'items_item_refund_in_usd', 'items_item_revenue', 'items_item_revenue_in_usd', 'items_item_variant', 'items_location_id', 'items_price', 'items_price_in_usd', 'items_promotion_id', 'items_promotion_name', 'items_quantity', 'platform', 'privacy_info_ads_storage', 'privacy_info_analytics_storage', 'privacy_info_uses_transient_token', 'stream_id', 'traffic_source_medium', 'traffic_source_name', 'traffic_source_source', 'user_first_touch_timestamp', 'user_id', 'user_ltv_currency', 'user_ltv_revenue', 'user_properties_key', 'user_properties_value', 'user_properties_value_double_value', 'user_properties_value_float_value', 'user_properties_value_int_value', 'user_properties_value_set_timestamp_micros', 'user_properties_value_string_value', 'user_pseudo_id']
    re_dimensions['is_metric']= False
    re_dimensions.loc[re_dimensions['name'].isin(list_met),'is_metric']=True
    re_dimensions['is_dimension']= True
    re_dimensions.loc[re_dimensions['name'].isin(list_met),'is_dimension']=False
    re_dimensions.loc[[6, 8, 9, 10, 11, 13, 14, 18, 20, 22, 24, 26, 29, 32, 34, 35, 46, 47, 48, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 65, 66, 67, 69, 71, 72, 75, 76, 78, 82],'description']= ['The type of device: Desktop, Tablet, or Mobile.', "The language setting of the user's browser or device. For example,\xa0English.", 'Manufacturer or branded name (examples: Samsung, HTC, Verizon, T-Mobile).', 'The branded device name (examples: Galaxy S10 or P30 Pro).', 'The mobile device model name (examples: iPhone X or SM-G950F).', 'The operating systems used by visitors to your app or website. Includes desktop and mobile operating systems such as Windows and Android.', "The operating system versions used by visitors to your website or app. For example, Android 10's version is 10, and iOS 13.5.1's version is 13.5.1.", 'The browsers used to view your website.', 'The sum of revenue from purchases minus refunded transaction revenue made in your app or site. Purchase revenue sums the revenue for these events:\xa0purchase,\xa0ecommerce_purchase,\xa0in_app_purchase,\xa0app_store_subscription_convert, and\xa0app_store_subscription_renew. Purchase revenue is specified by the\xa0value\xa0parameter in tagging.', 'The total refunded transaction revenues. Refund amount sums refunded revenue for the\xa0refund\xa0and\xa0app_store_refund\xa0events.', 'Shipping amount associated with a transaction. Populated by the\xa0shipping\xa0event parameter.', 'Tax amount associated with a transaction. Populated by the\xa0tax\xa0event parameter.', 'The ID of the ecommerce transaction.', 'The date of the event, formatted as YYYYMMDD.', 'Includes the subdomain and domain names of a URL; for example, the Host Name of www.example.com/contact.html is www.example.com.', 'The name of the event.', 'The city from which the user activity originated.', 'The continent from which the user activity originated. For example,\xa0Americas\xa0or\xa0Asia.', 'The country from which the user activity originated.', 'The geographic region from which the user activity originated, derived from their IP address.', 'The name or code of the affiliate (partner/vendor; if any) associated with an individual item. Populated by the\xa0affiliation\xa0item parameter.', 'Code for the order-level coupon.', 'The name of the item-promotion creative.', 'The name of the promotional creative slot associated with the item. This dimension can be specified in tagging by the\xa0creative_slot\xa0parameter at the event or item level. If the parameter is specified at both the event & item level, the item-level parameter is used.', 'Brand name of the item.', 'The hierarchical category in which the item is classified. For example, in Apparel/Mens/Summer/Shirts/T-shirts, Apparel is the item category.', 'The hierarchical category in which the item is classified. For example, in Apparel/Mens/Summer/Shirts/T-shirts, Mens is the item category 2.', 'The hierarchical category in which the item is classified. For example, in Apparel/Mens/Summer/Shirts/T-shirts, Summer is the item category 3.', 'The hierarchical category in which the item is classified. For example, in Apparel/Mens/Summer/Shirts/T-shirts, Shirts is the item category 4.', 'The hierarchical category in which the item is classified. For example, in Apparel/Mens/Summer/Shirts/T-shirts, T-shirts is the item category 5.', 'The ID of the item.', 'The ID of the item list.', 'The name of the item list.', 'The name of the item.', 'Item refund amount is the total refunded transaction revenue from items only. Item refund amount is the product of price and quantity for the\xa0refund\xa0event.', 'The total revenue from purchases minus refunded transaction revenue from items only. Item revenue is the product of its price and quantity. Item revenue excludes tax and shipping values; tax & shipping values are specified at the event and not item level.', 'The specific variation of a product. For example, XS, S, M, or L for size; or Red, Blue, Green, or Black for color. Populated by the\xa0item_variant\xa0parameter.', "The physical location associated with the item. For example, the physical store location. It's recommended to use the\xa0Google Place ID\xa0that corresponds to the associated item. A custom location ID can also be used. This field is populated in tagging by the\xa0location_id\xa0parameter in the items array.", 'The ID of the item promotion.', 'The name of the promotion for the item.', "The platform on which your app or website ran; for example, web, iOS, or Android. To determine a stream's type in a report, use both platform and streamId.", 'The numeric data stream identifier for your app or website.']
    re_dimensions['can_be_aggregated']= False
    re_dimensions.loc[[20, 22, 24, 67, 69],'can_be_aggregated']= True
    re_dimensions['data_type']= ['STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'INTEGER', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'INTEGER', 'STRING', 'STRING', 'INTEGER', 'INTEGER', 'RECORD', 'STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT', 'INTEGER', 'STRING', 'INTEGER', 'INTEGER', 'STRING', 'RECORD', 'STRING', 'STRING', 'STRING', 'RECORD', 'FLOAT', 'FLOAT', 'INTEGER', 'STRING', 'INTEGER', 'INTEGER', 'INTEGER', 'FLOAT', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT', 'STRING', 'STRING', 'FLOAT', 'FLOAT', 'STRING', 'STRING', 'INTEGER', 'STRING', 'INTEGER', 'INTEGER', 'STRING', 'INTEGER', 'STRING', 'STRING', 'STRING', 'INTEGER', 'STRING', 'STRING', 'FLOAT', 'INTEGER', 'RECORD', 'INTEGER', 'INTEGER', 'INTEGER', 'INTEGER', 'INTEGER', 'STRING']
    re_dimensions['is_geography_dependent']=False
    re_dimensions.loc[[46, 47, 48, 50, 72],'is_geography_dependent']= True
    re_dimensions['vendor']= 'Google'    
    re_dimensions['category']= 'GA4'
    re_dimensions['domain_id'] = domains[domains['domain_name'].str.contains('Media')]['domain_id'][0]

    Helper.write_dataframe_to_postgresql_in_chunks(re_dimensions, re_dimensions_table_name, chunksize=10000, connection_string=connection_string)

if __name__ == '__main__':
   populate_re_dimensions()
