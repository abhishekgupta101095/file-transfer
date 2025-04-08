import pandas as pd
import json
import scripts.Helper as Helper
import logging
import uuid

import scripts.dbConfig as db

import scripts.config as config

import ast

postgresPort=db.postgresPort
postgresHost=db.postgresHost
postgresDB=db.postgresDB
postgresUser=db.postgresUser
postgresPassword=db.postgresPassword

individual_3p_table_name=config.individual_3p_table_name
downloadedRawFilesLocation=config.locatPathToDownloadRawFiles


# Database connection parameters
db_params = {
    'host': postgresHost,
    'port': postgresPort,
    'database': postgresDB,
    'user': postgresUser,
    'password': postgresPassword
}

connection_string=f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["database"]}'

segment_file=f'{downloadedRawFilesLocation}Segments.csv'
demographic_file=f'{downloadedRawFilesLocation}AA_Demog.csv'
affinities_file=f'{downloadedRawFilesLocation}AA_Affinities.csv'


df_app_names = {
          "Num_Apps_beauty": ['Beauty1', 'Beauty2', 'Beauty3', 'Beauty4', 'Beauty5', 'Beauty6',
                                      'Beauty7', 'Beauty8'],
        "Num_Apps_books_and_reference": ['Amazon Kindle', 'Blinkist', 'Bookly', 'ComiXology', 'Scribd', 'Wattpad',
                                      'Goodreads', 'Inkitt'],
        "Num_Apps_dating": ['Tinder', 'Bumble', 'Hinge', 'Happn', 'Aisle', 'Badoo'],
        "Num_Apps_education": ['Vedantu', 'Vidyakul', 'Toppr', 'Doubtnut', 'Coursera', 'Indigolearn', 'Testbook',
                            'Unacademy', 'Simplilearn', 'Udemy'],
        "Num_Apps_entertainment": ['Netflix', 'Amazon Prime Video', 'Disney', 'TikTok', 'YouTube', 'Hulu'],
        "Num_Apps_food_and_drink": ['zomato', 'swiggy', 'foodpanda', 'domino', 'uber eats', 'doordarsh', 'starbucks',
                                 'subway'],
        "Num_Apps_health_and_fitness": ['MyFitnessPal', 'FitOn', 'forbeshealth', 'burn.fit', 'jefit'],
        "Num_Apps_house_and_home": ['NoBroker', 'Magicbricks', '99acres', 'Nestaway', 'Flatchat'],
        "Num_Apps_lifestyle": ['Pinterest', 'Google Home', 'SmartThings', 'H&M', 'Amazon Alexa'],
        "Num_Apps_music": ['Spotify', 'JioSaavn', 'Gaana', 'Wynk Music', 'Apple Music', 'Amazon Prime Music',
                        'Hungama', 'Coke Studio', 'SoundCloud'],
        "Num_Apps_parenting": ['TELL', 'Nanni AI', 'Pampers', 'Family360', 'Familysafe'],
        "Num_Apps_photo_and_video": ['Google Photos', 'KineMaster', 'Fimora', 'Splice', 'VivaVideo', 'Magisto',
                                   'InShot', 'CapCut'],
        "Num_Apps_shopping": ['flipcart', 'amazon', 'ebay', 'Myntra', 'Shopclues', 'tataCliq', 'PepperFry', 'Nykaa',
                           'Snapdeal', 'FirstCry'],
        "Num_Apps_social": ['twitter', 'x', 'facebook', 'instagram', 'linkedin', 'sharechat', 'whatsapp', 'Moj']
          }
   
   
# Function to update app names based on counts
def update_apps(row):
    updated_apps = {}
    row_dict = ast.literal_eval(row)
    for category, count in row_dict.items():
        if count > 0:
            apps = df_app_names[category][:count]
            updated_apps[category] = apps
        else:
            updated_apps[category] = []
    return updated_apps

def individual_process_3p():
    df_segement = pd.read_csv(segment_file,usecols=["segment_id","Segment"])
    df_social_col = pd.read_csv('/opt/spark-data/raw/3p_social.csv')
    dict_segment = df_segement.set_index('Segment')['segment_id'].to_dict()
    
     
     
     
    df_demo = pd.read_csv(demographic_file,usecols=['Individual_ID','Age Group','Education Level','Gender',
                                                                                         'Home Owner Status','Household Income','Individual Income',
                                                                                         'Net Worth','first_name','last_name','phone_number',
                                                                                          'email'])
                                                                                       
     
    df_affinity = pd.read_csv(affinities_file,usecols=['Individual_ID','Segment','Married',
                                                       'NumberOfAdultsInHH','NumberOfChildren',
                                                       'PremiumCardHolder' ])
                                                                                          
    df_full = df_demo.merge(df_affinity, on='Individual_ID')
     
    # # Replace app categories with IDs in Table 2
    # df_demo['Segment'] = df_demo['Segment'].map(segmeent_map)
     
    
     #df_visits with segment id
    df_full['segment_id'] = df_full['Segment'].map(dict_segment)
    df_full.drop(['Segment'], axis=1, inplace=True)
    
    
    df_occupation = pd.read_csv(affinities_file,usecols=['Individual_ID','Business Owner',
                                                                                            'Civil Service','Computer','Exec/Upper Mgmt',
                                                                                              'Health Services','Homemaker','Middle Management','Military Personnel',
                                                                                              'Nurse','Part Time','Professional','Retired','Secretary','Student','Teacher',
                                                                                              'WFH_occupations'])
                                                                                         
    df_occupation.to_csv('occ.csv', index=False)  # Optional: Save to a CSV file
    df_occupation = df_occupation.melt(id_vars=['Individual_ID'], var_name='occupation', value_name='count')
    # Filter rows where 'count' is 1 and keep relevant columns
    df_occupation = df_occupation[df_occupation['count'] == 1][['Individual_ID', 'occupation']]
     
     
    df_urbancity = pd.read_csv(affinities_file,usecols=['Individual_ID','Urbanicity_rural','Urbanicity_suburban','Urbanicity_urban'])
    df_urbancity.to_csv('urbancity.csv', index=False)  # Optional: Save to a CSV file                                                                                       
    df_urbancity = df_urbancity.melt(id_vars=['Individual_ID'], var_name='urbancity', value_name='count')
    # # Filter rows where 'count' is 1 and keep relevant columns
    df_urbancity = df_urbancity[df_urbancity['count'] == 1][['Individual_ID', 'urbancity']]

    df_collar_worker = pd.read_csv(affinities_file,usecols=['Individual_ID','Blue Collar Worker','White Collar Worker'])
                                                                                         
    df_collar_worker.to_csv('coll.csv', index=False)  # Optional: Save to a CSV file
    df_collar_worker = df_collar_worker.melt(id_vars=['Individual_ID'], var_name='collar_worker', value_name='count')
    # Filter rows where 'count' is 1 and keep relevant columns
    df_collar_worker = df_collar_worker[df_collar_worker['count'] == 1][['Individual_ID', 'collar_worker']]
     
    df_final_urban=df_full.merge(df_urbancity, on='Individual_ID')     
    df_occupation=df_final_urban.merge(df_occupation, on='Individual_ID')
    df_final=df_occupation.merge(df_collar_worker, on='Individual_ID')

    df_final['location'] = ""
    df_final['cookie_id'] = "AYQEVnDKrdst"
    df_final['ip_address'] = "192.158.1.38"
    df_final['device_id'] = "00000000-89ABCDEF-01234567-89ABCDEF"
    #df_final['load_id'] = 1
   
    ############################################
    # print(df_final.columns)
    df_apps = pd.read_csv(affinities_file,usecols=['Individual_ID','Num_Apps_beauty','Num_Apps_books_and_reference','Num_Apps_dating','Num_Apps_education','Num_Apps_entertainment',
                                                                                          'Num_Apps_food_and_drink','Num_Apps_health_and_fitness','Num_Apps_house_and_home','Num_Apps_lifestyle','Num_Apps_music','Num_Apps_parenting',
                                                                                          'Num_Apps_photo_and_video','Num_Apps_shopping','Num_Apps_social'])
    
    # Convert columns to a dictionary
    apps_dict = df_apps.iloc[:, 1:].to_dict(orient='records')[0]
    print(apps_dict)
     
    # Create a new 'Visits' column with the dictionary
    df_apps['no_of_apps'] = df_apps.apply(lambda row: json.dumps({brand.replace("# Num_Apps ", ''): row[brand] for brand in apps_dict}), axis=1)
    
    # Specify columns to remove
    columns_to_remove = ['Num_Apps_beauty','Num_Apps_books_and_reference','Num_Apps_dating','Num_Apps_education','Num_Apps_entertainment',
                                                                                          'Num_Apps_food_and_drink','Num_Apps_health_and_fitness','Num_Apps_house_and_home','Num_Apps_lifestyle','Num_Apps_music','Num_Apps_parenting',
                                                                                          'Num_Apps_photo_and_video','Num_Apps_shopping','Num_Apps_social']
    
    # Remove columns using `drop`
    df_apps = df_apps.drop(columns_to_remove, axis=1)  # axis=1 for columns
    
    # Apply the function to update the DataFrame
    df_apps['no_of_apps'] = df_apps['no_of_apps'].apply(update_apps)
    df_apps["no_of_apps"]=df_apps["no_of_apps"].apply(str)
    print(df_apps)
     ############################################
    
    # df_final_appcount=df_final.merge(df_apps, on='Individual_ID')
    # df_final_appcount.to_csv("test101.csv", index=False)  # Optional: Save to a CSV file
    
    df_affinity = pd.read_csv(affinities_file,usecols=['Individual_ID','Beauty_affinity','Catalog_affinity','Cooking_affinity','Discount_affinity','Education_affinity',
                                                                                          'Entertainment_affinity','Fashion_affinity','Gambling_affinity','Gardening_affinity','HealthyLiving_affinity','Investment_affinity',
                                                                                          'Lifestyle_luxury_affinity','Movies_affinity','Music_affinity','Outdoors_affinity',
                                                                                           'Subscription_affinity','Travel_affinity','WomenProductBuyer_affinity'])
                                                                                            
    
    # Select relevant columns
    df_affinity_id = df_affinity[['Individual_ID']]
     
    # Create the 'visita' column by filtering city visit columns and converting to lists
    df_affinity_id['affinity'] = df_affinity.filter(like='_affinity').apply(lambda x: x.index[x == 1].tolist(), axis=1)
    
    # df_affinity_id.to_csv("test101.csv", index=False)  # Optional: Save to a CSV file
    
    # Remove 'visit_' prefix from city names
    df_affinity_id['affinity'] = df_affinity_id['affinity'].apply(lambda x: [affinity[:-9] for affinity in x])
    
    print('Affinity column is:')
    print(df_affinity_id['affinity'])
    print(df_affinity_id)
    ############################################
    
    
     
    df_visits = pd.read_csv(affinities_file,usecols=["Individual_ID","# Visits to Macy's",
                                                                                                   "# Visits to Nordstorm","# Visits to Sephora",
                                                                                                      "# Visits to Target","# Visits to Ulta Beauty"
                                                                                                      ])
                                                                                           
     
    df_visits_dict = df_visits.iloc[:, 1:].to_dict(orient='records')[0]
    
    df_visits['visits'] = df_visits.apply(lambda row: json.dumps({brand.replace("Num_Browse_ ", ''): row[brand] for brand in df_visits_dict}), axis=1)
     
    
    df_visits = df_visits.drop(["# Visits to Macy's","# Visits to Nordstorm","# Visits to Sephora","# Visits to Target","# Visits to Ulta Beauty"], axis=1)  # axis=1 for columns
     
    
    ############################################
    
    #### Montly Visit 
    df_monthlyVisit = pd.read_csv(affinities_file,usecols=['Individual_ID','Monthly_Visit_Freq_Macys',
                                                                            'Monthly_Visit_Freq_Nordstorm','Monthly_Visit_Freq_Sephora',
                                                                            'Monthly_Visit_Freq_Target','Monthly_Visit_Freq_Ulta_Beauty',
                                                                                          ])
                                                                         
    
    # Convert columns to a dictionary
    monthlyvisit_dict = df_monthlyVisit.iloc[:, 1:].to_dict(orient='records')[0]
    # print(monthlyvisit_dict)
     
    # Create a new 'Visits' column with the dictionary
    df_monthlyVisit['monthly_visits'] = df_monthlyVisit.apply(lambda row: json.dumps({brand.replace("# Monthly_Visit_Freq_ ", ''): row[brand] for brand in monthlyvisit_dict}), axis=1)
    
    # Specify columns to remove
    columns_to_remove = ['Monthly_Visit_Freq_Macys','Monthly_Visit_Freq_Nordstorm','Monthly_Visit_Freq_Sephora',
                                                                            'Monthly_Visit_Freq_Target','Monthly_Visit_Freq_Ulta_Beauty']
    
    # Remove columns using `drop`
    df_monthlyVisit = df_monthlyVisit.drop(columns_to_remove, axis=1)  # axis=1 for columns
    
    print(df_monthlyVisit)
    #### Weekday Visit
    df_weekdayVisit = pd.read_csv(affinities_file,usecols=['Individual_ID','Num_weekday_visit_Macys',
                                                                            'Num_weekday_visit_Nordstorm','Num_weekday_visit_Sephora',
                                                                            'Num_weekday_visit_Target','Num_weekday_visit_Ulta_Beauty',
                                                                                          ])
                                                                             
    # Convert columns to a dictionary
    weekdayvisit_dict = df_weekdayVisit.iloc[:, 1:].to_dict(orient='records')[0]
    # print(weekdayvisit_dict)
     
    # Create a new 'Visits' column with the dictionary
    df_weekdayVisit['weekday_visits'] = df_weekdayVisit.apply(lambda row: json.dumps({brand.replace("# Num_weekday_visit_ ", ''): row[brand] for brand in weekdayvisit_dict}), axis=1)
    
    # Specify columns to remove
    columns_to_remove = ['Num_weekday_visit_Macys','Num_weekday_visit_Nordstorm','Num_weekday_visit_Sephora',
                          'Num_weekday_visit_Target','Num_weekday_visit_Ulta_Beauty']
    
    # Remove columns using `drop`
    df_weekdayVisit = df_weekdayVisit.drop(columns_to_remove, axis=1)  # axis=1 for columns
    print(df_weekdayVisit)
    
    
    ##### Weekend Visit
    df_weekendVisit = pd.read_csv(affinities_file,usecols=['Individual_ID','Num_weekend_visit_Macys',
                                                                            'Num_weekend_visit_Nordstorm','Num_weekend_visit_Sephora',
                                                                            'Num_weekend_visit_Target','Num_weekend_visit_Ulta_Beauty',
                                                                                          ])
    
    # # Convert columns to a dictionary
    weekendvisit_dict = df_weekendVisit.iloc[:, 1:].to_dict(orient='records')[0]
    
     
    # Create a new 'Visits' column with the dictionary
    df_weekendVisit['weekend_visits'] = df_weekendVisit.apply(lambda row: json.dumps({brand.replace("# Num_weekend_visit_ ", ''): row[brand] for brand in weekendvisit_dict}), axis=1)
    
    # Specify columns to remove
    columns_to_remove = ['Num_weekend_visit_Macys','Num_weekend_visit_Nordstorm','Num_weekend_visit_Sephora',
                        'Num_weekend_visit_Target','Num_weekend_visit_Ulta_Beauty']
    
    # Remove columns using `drop`
    df_weekendVisit = df_weekendVisit.drop(columns_to_remove, axis=1)  # axis=1 for columns
    print(df_weekendVisit.columns)
    
    
    
    ############################################
    # num_domain_visits
    df_num_domain_visit = pd.read_csv(affinities_file,usecols=["Individual_ID","Num_Days_domain_visitapps_&_online_tools","Num_Days_domain_visitgyms_&_health_clubs","Num_Days_domain_visitonline_image_galleries","Num_Days_domain_visitshopping_apparel",
    "Num_Days_domain_visitshopping_apparel_athletic_apparel","Num_Days_domain_visitshopping_apparel_casual_apparel",
    "Num_Days_domain_visitshopping_apparel_children's_clothing","Num_Days_domain_visitshopping_apparel_clothing_accessories","Num_Days_domain_visitshopping_apparel_costumes","Num_Days_domain_visitshopping_apparel_eyewear",	"Num_Days_domain_visitshopping_apparel_footwear","Num_Days_domain_visitshopping_apparel_formal_wear","Num_Days_domain_visitshopping_apparel_headwear","Num_Days_domain_visitshopping_apparel_outerwear",	"Num_Days_domain_visitshopping_apparel_services","Num_Days_domain_visitshopping_apparel_sleepwear","Num_Days_domain_visitshopping_apparel_suits_&_business_attire","Num_Days_domain_visitshopping_apparel_swimwear",	"Num_Days_domain_visitshopping_apparel_undergarments","Num_Days_domain_visitshopping_apparel_uniforms_&_workwear","Num_Days_domain_visitshopping_apparel_women's_clothing",	"Num_Days_domain_visitshopping_consumer_resources_coupons_&_discount_offers","Num_Days_domain_visitshopping_consumer_resources_product_reviews_&_price_comparisons","Num_Days_domain_visitshopping_entertainment_media",	"Num_Days_domain_visitshopping_green_&_eco_friendly_shopping","Num_Days_domain_visitvideo-online_video","Num_Days_domain_visitwebcams_&_virtual_tours"])
    
    # Convert columns to a dictionary
    num_domain_visit_dict = df_num_domain_visit.iloc[:, 1:].to_dict(orient='records')[0]
    
     
    # Create a new 'Visits' column with the dictionary
    df_num_domain_visit['num_days_domain_visits'] = df_num_domain_visit.apply(lambda row: json.dumps({brand.replace("Num_Days_domain_visit ", ''): row[brand] for brand in num_domain_visit_dict}), axis=1)
    
    # Specify columns to remove
    columns_to_remove = ["Num_Days_domain_visitapps_&_online_tools","Num_Days_domain_visitgyms_&_health_clubs","Num_Days_domain_visitonline_image_galleries","Num_Days_domain_visitshopping_apparel",
    "Num_Days_domain_visitshopping_apparel_athletic_apparel","Num_Days_domain_visitshopping_apparel_casual_apparel",
    "Num_Days_domain_visitshopping_apparel_children's_clothing","Num_Days_domain_visitshopping_apparel_clothing_accessories","Num_Days_domain_visitshopping_apparel_costumes","Num_Days_domain_visitshopping_apparel_eyewear",	"Num_Days_domain_visitshopping_apparel_footwear","Num_Days_domain_visitshopping_apparel_formal_wear","Num_Days_domain_visitshopping_apparel_headwear","Num_Days_domain_visitshopping_apparel_outerwear",	"Num_Days_domain_visitshopping_apparel_services","Num_Days_domain_visitshopping_apparel_sleepwear","Num_Days_domain_visitshopping_apparel_suits_&_business_attire","Num_Days_domain_visitshopping_apparel_swimwear",	"Num_Days_domain_visitshopping_apparel_undergarments","Num_Days_domain_visitshopping_apparel_uniforms_&_workwear","Num_Days_domain_visitshopping_apparel_women's_clothing",	"Num_Days_domain_visitshopping_consumer_resources_coupons_&_discount_offers","Num_Days_domain_visitshopping_consumer_resources_product_reviews_&_price_comparisons","Num_Days_domain_visitshopping_entertainment_media",	"Num_Days_domain_visitshopping_green_&_eco_friendly_shopping","Num_Days_domain_visitvideo-online_video","Num_Days_domain_visitwebcams_&_virtual_tours"]
    
    # Remove columns using `drop`
    df_num_domain_visit = df_num_domain_visit.drop(columns_to_remove, axis=1)  # axis=1 for columns
    print(df_num_domain_visit)
    
    
    
    
    # num_domain_surf
    df_num_domain_surf = pd.read_csv(affinities_file,usecols=["Individual_ID","Num_DomainSurf_apps_&_online_tools","Num_DomainSurf_gyms_&_health_clubs","Num_DomainSurf_online_image_galleries",
                                                                                                            "Num_DomainSurf_shopping_apparel","Num_DomainSurf_shopping_apparel_athletic_apparel",	"Num_DomainSurf_shopping_apparel_casual_apparel","Num_DomainSurf_shopping_apparel_children's_clothing","Num_DomainSurf_shopping_apparel_clothing_accessories",
                                                                                                            "Num_DomainSurf_shopping_apparel_costumes",	"Num_DomainSurf_shopping_apparel_eyewear","Num_DomainSurf_shopping_apparel_footwear","Num_DomainSurf_shopping_apparel_formal_wear","Num_DomainSurf_shopping_apparel_headwear",
                                                                                                            "Num_DomainSurf_shopping_apparel_outerwear",	"Num_DomainSurf_shopping_apparel_services","Num_DomainSurf_shopping_apparel_sleepwear","Num_DomainSurf_shopping_apparel_suits_&_business_attire","Num_DomainSurf_shopping_apparel_swimwear",	"Num_DomainSurf_shopping_apparel_undergarments","Num_DomainSurf_shopping_apparel_uniforms_&_workwear","Num_DomainSurf_shopping_apparel_women's_clothing","Num_DomainSurf_shopping_consumer_resources_coupons_&_discount_offers","Num_DomainSurf_shopping_green_&_eco-friendly_shopping",
                                                                                                           "Num_DomainSurf_shopping_entertainment_media","Num_DomainSurf_arts_&_entertainment_tv_&_video_online_video","Num_DomainSurf_webcams_&_virtual_tours"                                                                                                            
    ])       
    # Convert columns to a dictionary
    num_domain_surf_dict = df_num_domain_surf.iloc[:, 1:].to_dict(orient='records')[0]
    # print(num_domain_surf_dict)
     
    # Create a new 'Visits' column with the dictionary
    df_num_domain_surf['num_days_domain_surf'] = df_num_domain_surf.apply(lambda row: json.dumps({brand.replace("Num_DomainSurf_ ", ''): row[brand] for brand in num_domain_surf_dict}), axis=1)
  
    # Specify columns to remove
    columns_to_remove = ["Num_DomainSurf_apps_&_online_tools","Num_DomainSurf_gyms_&_health_clubs","Num_DomainSurf_online_image_galleries",
                                                                                                            "Num_DomainSurf_shopping_apparel","Num_DomainSurf_shopping_apparel_athletic_apparel","Num_DomainSurf_shopping_apparel_casual_apparel","Num_DomainSurf_shopping_apparel_children's_clothing","Num_DomainSurf_shopping_apparel_clothing_accessories",
                                                                                                            "Num_DomainSurf_shopping_apparel_costumes",	"Num_DomainSurf_shopping_apparel_eyewear","Num_DomainSurf_shopping_apparel_footwear","Num_DomainSurf_shopping_apparel_formal_wear","Num_DomainSurf_shopping_apparel_headwear",
                                                                                                            "Num_DomainSurf_shopping_apparel_outerwear",	"Num_DomainSurf_shopping_apparel_services","Num_DomainSurf_shopping_apparel_sleepwear","Num_DomainSurf_shopping_apparel_suits_&_business_attire","Num_DomainSurf_shopping_apparel_swimwear","Num_DomainSurf_shopping_apparel_undergarments","Num_DomainSurf_shopping_apparel_uniforms_&_workwear","Num_DomainSurf_shopping_apparel_women's_clothing","Num_DomainSurf_shopping_consumer_resources_coupons_&_discount_offers",	"Num_DomainSurf_shopping_green_&_eco-friendly_shopping",
                                                                                                            "Num_DomainSurf_shopping_entertainment_media","Num_DomainSurf_arts_&_entertainment_tv_&_video_online_video","Num_DomainSurf_webcams_&_virtual_tours"]
    
    # Remove columns using `drop`
    df_num_domain_surf = df_num_domain_surf.drop(columns_to_remove, axis=1)  # axis=1 for columns
    
    
    print(df_num_domain_surf)
    
    ############################################
    df_avg_browsing_perday = pd.read_csv(affinities_file,usecols=["Individual_ID","Avg_Browsing_Perday_apps_&_online_tools","Avg_Browsing_Perday_gyms_&_health_clubs","Avg_Browsing_Perday_online_image_galleries","Avg_Browsing_Perday_shopping_apparel","Avg_Browsing_Perday_shopping_apparel_athletic_apparel","Avg_Browsing_Perday_shopping_apparel_casual_apparel","Avg_Browsing_Perday_shopping_apparel_children's_clothing","Avg_Browsing_Perday_shopping_apparel_clothing_accessories","Avg_Browsing_Perday_shopping_apparel_costumes","Avg_Browsing_Perday_shopping_apparel_eyewear","Avg_Browsing_Perday_shopping_apparel_footwear","Avg_Browsing_Perday_shopping_apparel_formal_wear","Avg_Browsing_Perday_shopping_apparel_headwear","Avg_Browsing_Perday_shopping_apparel_outerwear","Avg_Browsing_Perday_shopping_apparel_services","Avg_Browsing_Perday_shopping_apparel_sleepwear","Avg_Browsing_Perday_shopping_apparel_suits_&_business_attire","Avg_Browsing_Perday_shopping_apparel_swimwear","Avg_Browsing_Perday_shopping_apparel_undergarments","Avg_Browsing_Perday_shopping_apparel_uniforms_&_workwear","Avg_Browsing_Perday_shopping_apparel_women's_clothing","Avg_Browsing_Perday_shopping_consumer_resources_coupons_&_discount_offers","Avg_Browsing_Perday_shopping_entertainment_media","Avg_Browsing_Perday_shopping_green_&_eco_friendly_shopping","Avg_Browsing_Perday_video-online_video","Avg_Browsing_Perday_webcams_&_virtual_tours"])
    
     
    # Convert columns to a dictionary
    avg_browsing_per_day_dict = df_avg_browsing_perday.iloc[:, 1:].to_dict(orient='records')[0]
    print(avg_browsing_per_day_dict)
     
    # # Create a new 'Visits' column with the dictionary
    df_avg_browsing_perday['avg_browsing_per_day'] = df_avg_browsing_perday.apply(lambda row: json.dumps({brand.replace("# Avg_Browsing_Perday_ ", ''): row[brand] for brand in avg_browsing_per_day_dict}), axis=1)
     
    # Specify columns to remove
    columns_to_remove = ["Avg_Browsing_Perday_apps_&_online_tools","Avg_Browsing_Perday_gyms_&_health_clubs","Avg_Browsing_Perday_online_image_galleries","Avg_Browsing_Perday_shopping_apparel","Avg_Browsing_Perday_shopping_apparel_athletic_apparel","Avg_Browsing_Perday_shopping_apparel_casual_apparel","Avg_Browsing_Perday_shopping_apparel_children's_clothing","Avg_Browsing_Perday_shopping_apparel_clothing_accessories","Avg_Browsing_Perday_shopping_apparel_costumes","Avg_Browsing_Perday_shopping_apparel_eyewear","Avg_Browsing_Perday_shopping_apparel_footwear","Avg_Browsing_Perday_shopping_apparel_formal_wear","Avg_Browsing_Perday_shopping_apparel_headwear","Avg_Browsing_Perday_shopping_apparel_outerwear","Avg_Browsing_Perday_shopping_apparel_services","Avg_Browsing_Perday_shopping_apparel_sleepwear","Avg_Browsing_Perday_shopping_apparel_suits_&_business_attire","Avg_Browsing_Perday_shopping_apparel_swimwear","Avg_Browsing_Perday_shopping_apparel_undergarments","Avg_Browsing_Perday_shopping_apparel_uniforms_&_workwear","Avg_Browsing_Perday_shopping_apparel_women's_clothing","Avg_Browsing_Perday_shopping_consumer_resources_coupons_&_discount_offers","Avg_Browsing_Perday_shopping_entertainment_media","Avg_Browsing_Perday_shopping_green_&_eco_friendly_shopping","Avg_Browsing_Perday_video-online_video","Avg_Browsing_Perday_webcams_&_virtual_tours"]
    # Remove columns using `drop`
    df_avg_browsing_perday = df_avg_browsing_perday.drop(columns_to_remove, axis=1)  # axis=1 for columns
    print(df_avg_browsing_perday)
    
    
    df_num_browse = pd.read_csv(affinities_file,usecols=["Individual_ID","Num_Browse_apps_&_online_tools","Num_Browse_gyms_&_health_clubs","Num_Browse_online_image_galleries","Num_Browse_shopping_apparel","Num_Browse_shopping_apparel_athletic_apparel","Num_Browse_shopping_apparel_casual_apparel","Num_Browse_shopping_apparel_children's_clothing","Num_Browse_shopping_apparel_clothing_accessories","Num_Browse_shopping_apparel_costumes","Num_Browse_shopping_apparel_eyewear","Num_Browse_shopping_apparel_footwear","Num_Browse_shopping_apparel_formal_wear","Num_Browse_shopping_apparel_headwear","Num_Browse_shopping_apparel_outerwear","Num_Browse_shopping_apparel_services","Num_Browse_shopping_apparel_sleepwear","Num_Browse_shopping_apparel_suits_&_business_attire","Num_Browse_shopping_apparel_swimwear","Num_Browse_shopping_apparel_undergarments","Num_Browse_shopping_apparel_uniforms_&_workwear","Num_Browse_shopping_apparel_women's_clothing","Num_Browse_shopping_consumer_resources_coupons_&_discount_offers","Num_Browse_shopping_consumer_resources_product_reviews_&_price_comparisons","Num_Browse_shopping_entertainment_media","Num_Browse_shopping_green_&_eco_friendly_shopping","Num_Browse_video-online_video","Num_Browse_webcams_&_virtual_tours"])
    
     
    df_num_browse_dict = df_num_browse.iloc[:, 1:].to_dict(orient='records')[0]
    
     
    # Create a new 'Visits' column with the dictionary
    df_num_browse['num_browse'] = df_num_browse.apply(lambda row: json.dumps({brand.replace("Num_Browse_ ", ''): row[brand] for brand in df_num_browse_dict}), axis=1)
    print(df_num_browse.columns)
    print(df_num_browse.head(10))
    # Specify columns to remove
    columns_to_remove = ["Num_Browse_apps_&_online_tools","Num_Browse_gyms_&_health_clubs","Num_Browse_online_image_galleries","Num_Browse_shopping_apparel","Num_Browse_shopping_apparel_athletic_apparel","Num_Browse_shopping_apparel_casual_apparel","Num_Browse_shopping_apparel_children's_clothing","Num_Browse_shopping_apparel_clothing_accessories","Num_Browse_shopping_apparel_costumes","Num_Browse_shopping_apparel_eyewear","Num_Browse_shopping_apparel_footwear","Num_Browse_shopping_apparel_formal_wear","Num_Browse_shopping_apparel_headwear","Num_Browse_shopping_apparel_outerwear","Num_Browse_shopping_apparel_services","Num_Browse_shopping_apparel_sleepwear","Num_Browse_shopping_apparel_suits_&_business_attire","Num_Browse_shopping_apparel_swimwear","Num_Browse_shopping_apparel_undergarments","Num_Browse_shopping_apparel_uniforms_&_workwear","Num_Browse_shopping_apparel_women's_clothing","Num_Browse_shopping_consumer_resources_coupons_&_discount_offers","Num_Browse_shopping_consumer_resources_product_reviews_&_price_comparisons","Num_Browse_shopping_entertainment_media","Num_Browse_shopping_green_&_eco_friendly_shopping","Num_Browse_video-online_video","Num_Browse_webcams_&_virtual_tours"]
    # Remove columns using `drop`
    df_num_browse = df_num_browse.drop(columns_to_remove, axis=1)  # axis=1 for columns
    print(df_num_browse)
    
    
    ############################################
    
    df_final = df_final.merge(df_apps, on='Individual_ID')
    df_final = df_final.merge(df_affinity_id, on='Individual_ID')
    df_final=df_final.merge(df_monthlyVisit, on='Individual_ID')
    df_final=df_final.merge(df_weekdayVisit, on='Individual_ID')
    df_final=df_final.merge(df_weekendVisit, on='Individual_ID')
    df_final=df_final.merge(df_num_domain_visit, on='Individual_ID')
    df_final=df_final.merge(df_num_domain_surf, on='Individual_ID')
    df_final=df_final.merge(df_avg_browsing_perday, on='Individual_ID')
    df_final=df_final.merge(df_num_browse, on='Individual_ID')
    df_final=df_final.merge(df_visits, on='Individual_ID')
    print(df_final.columns)
    print(df_avg_browsing_perday.columns)
    
    
    df_final.rename(columns = {'Individual_ID':'cid'}, inplace = True)
    df_final.rename(columns = {'Age Group':'age_group'}, inplace = True)
    df_final.rename(columns = {'Education Level':'education_level'}, inplace = True) 
    df_final.rename(columns = {'Gender':'gender'}, inplace = True)
    df_final.rename(columns = {'Home Owner Status':'home_owner_status'}, inplace = True)
    df_final.rename(columns = {'Household Income':'household_income'}, inplace = True)  
    
    df_final.rename(columns = {'Individual Income':'individual_income'}, inplace = True) 
    df_final.rename(columns = {'Net Worth':'net_worth'}, inplace = True) 
    df_final.rename(columns = {'phone_number':'phone'}, inplace = True) 
    df_final.rename(columns = {'White Collar Worker':'collar_worker'}, inplace = True) 
    df_final.rename(columns = {'Married':'married'}, inplace = True) 
    df_final.rename(columns = {'NumberOfAdultsInHH':'no_of_adults_hh'}, inplace = True) 
    df_final.rename(columns = {'NumberOfChildren':'no_of_children_hh'}, inplace = True) 
    df_final.rename(columns = {'PremiumCardHolder':'premium_card_holder'}, inplace = True) 
    df_final.rename(columns = {'urbancity':'urbanicity'}, inplace = True) 
    df_final.rename(columns = {'occupation':'occupation'}, inplace = True) 
    df_final.rename(columns = {'num_days_domain_surf':'num_domain_surf'}, inplace = True) 
    
    df_final = df_final.drop('location', axis=1)  # axis=1 for columns
    
    num_rows = df_final.iloc[:].values.shape[0]
    uuid_data = {'load_id': [str(uuid.uuid4()) for _ in range(num_rows)]}
    uuid_df = pd.DataFrame(uuid_data)
    df_final = pd.concat([df_final, uuid_df], axis=1, ignore_index=False)
    df_final = df_final.merge(df_social_col, on='cid')

    
    # df_final.to_csv("3p_indi.csv", index=False)  # Optional: Save to a CSV file
    Helper.write_dataframe_to_postgresql_in_chunks(df_final,individual_3p_table_name, chunksize=10000, connection_string=connection_string)
