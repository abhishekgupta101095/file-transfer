##S3 Connection Details####
bucketName='re-dev-udm'
s3DictionaryPrefix='CDI/metadatafiles/raw/'
s3DictionaryProcessedPath='CDI/metadatafiles/processed/'
s3DictionaryFailedPath='CDI/metadatafiles/failed/'
s3Delimiter='/'
awsConnId="aws_udm"
aws_access_key_id=''
aws_secret_access_key=''
aws_region='eu-west-2'
############End of S3 Connection Detail####


locatPathToDownloadDictionaryFiles='/opt/spark-data/dictionary/'
filePatternToLoadDictionaryFiles=['web', 'apps', 'poi']
webDictCols=["id","category","tier1","tier2"]
#webDictCatTableName="\"3p_web_categories_meta\""
webDictCatTableName="3p_web_categories_meta"
appsDictCols=["category","description","id"]
appsDictCatTableName="3p_apps_meta"
poiDictCols=["category","subcategory","location","id"]
poiDictCatTableName="3p_poi_meta"
dataLoadMetadataTableName="data_load_metadata"

##S3 Configuration for 3p individual table
locatPathToDownloadRawFiles='/opt/spark-data/raw/'
s3RawFilesPrefix='CDI/data/raw/'
s3RawFilesFailed='CDI/data/failed/'

individual_3p_table_name="3p_individual"
segment_demo_insights_table_name="segment_demo_insights"

s3_processed_path='CDI/data/processed/'

segment_profile_web_table_name="segment_profile_web"
segment_profile_poi_table_name="segment_profile_poi"
segment_profile_app_table_name="segment_profile_app"

audience_analytics_table_name="audience_analytics"
audience_demo_insights_table_name="audience_demo_insights"
audience_profile_poi_table_name="audience_profile_poi"
audience_profile_app_table_name="audience_profile_app"
aa_social_view_table_name ='aa_social_view'
aa_web_view_table_name = 'aa_web_view'

idr_analytics_table_name="idr_analytics"
idr_demo_insights_table_name="idr_demo_insights"
idr_profile_poi_table_name="idr_profile_poi"
idr_profile_app_table_name="idr_profile_app"
idr_web_view_table_name="idr_web_view"
idr_social_view_table_name='idr_social_view'
idr_demo_summary_table_name="idr_demo_summary"

attribute_configs="/opt/airflow/dags/scripts/attribute_config.json"
dataQualityJsonFile='/opt/airflow/dags/scripts/config.json'

social_meta_table_name='3p_social_meta'
socialDictCols=['category1','category2','category3','category6']
segment_profile_social_table_name='segment_profile_social'
domains_table_name = "domains"
re_dimensions_table_name = "re_dimensions"
tokenization_required=False
token=''
files_with_columns_to_tokenize={'/opt/spark-data/raw/AA_Demog.csv':['email','phone_number']}
tokenization_url='https://172.71.71.235:8200/v1/udm/encryption/udm_key'


##IDR invoke configuration
containername_idrinvoke='amap_postgres_13'
database_idrinvoke='udm_db'
table_idrinvoke='3p_individual'
columns_idrinvoke='first_name,last_name,phone,email,cookie_id,device_id'
bucketName_idrinvoke=bucketName
aws_access_key_id_idrinvoke=aws_access_key_id
aws_secret_access_key_idrinvoke=aws_secret_access_key
s3location_idrinvoke='CDI/persistent_id/'
url_idrinvoke='http://localhost:5000/jobcompletion_notification'
job_status_table_name='jobmetadata'


individual_geographics_view_table_name='audience_geographics_view'
segment_geographics_view_table_name="segment_geographics_view"

demoggraphicsfilename='AA_Demog.csv'

#####IDR mapping job
idr_mappingfile_bucketName='re-dev-idr-mappingfiles'
localPathToDownloadIDRMappingFiles='/opt/spark-data/idrmappingfiles/'
idr_mappingFileS3Prefix='CDI/raw/'
s3_processed_path_idr_mapping='CDI/processed/'

cidCol='cid'
persistentIDCol='persistentid'
