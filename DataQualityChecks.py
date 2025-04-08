import json 
import pandas as pd
import scripts.config as config
import boto3
from datetime import datetime
import os
import phonenumbers
import pytz
import numpy as np
import usaddress

dataQualityJsonFile = config.dataQualityJsonFile

s3RawFilesPrefix = config.s3RawFilesPrefix
locatPathToDownloadRawFiles = config.locatPathToDownloadRawFiles
locatPathToDownloadDictionaryFiles = config.locatPathToDownloadDictionaryFiles
s3DictionaryPrefix=config.s3DictionaryPrefix
BUCKET_NAME=config.bucketName
s3DictionaryProcessedPath = config.s3DictionaryProcessedPath
s3_processed_path = config.s3_processed_path
s3DictionaryFailedPath = config.s3DictionaryFailedPath
s3RawFilesFailed = config.s3RawFilesFailed

aws_access_key_id=config.aws_access_key_id
aws_secret_access_key=config.aws_secret_access_key
aws_region=config.aws_region

def read_config(config_file_path):
    try:
        with open(config_file_path, 'r') as config_file:
            config = json.load(config_file)
        return config
    except Exception as e:
        print(f"Error reading config JSON: {str(e)}")

def read_data(source_location, filename,encoding):
  """
  Reads a CSV file into a DataFrame.

  Args:
      source_location: The path to the file, including directory and filename.
      filename: The filename of the file.

  Returns:
      A pandas DataFrame containing the data from the file.
  """  
  # Check the file extension to determine the read method
  if filename.endswith(".csv"):
    try:
      df = pd.read_csv(source_location + filename,encoding=encoding)
      return df
    except pd.errors.EmptyDataError:
      return None
  else:
    raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

def calculate_fill_rate(df, column):
  """
    Calculates the fill rate of a given column in a dataframe.
  Args:
      df: The dataframe to check.
      column: The name of the column to check.
  Returns:
      The fill rate of the specified column as a float between 0 and 1.
  Raises:
      ValueError: If the column does not exist in the dataframe.
  """
  if column not in df.columns:
    raise ValueError(f"Column '{column}' not found in DataFrame.")
  
  return (df[column].count() / len(df))

def completeness_check(df,columns_to_check,fill_rate_threshold=0.7,final_result=''):
 """
 Checks whetehr mandatory columns exists in the dataframe and calculate the fill rate of specified columns in a dataframe with error handling.

 Args:
     df: The dataframe to check.
     columns_to_check: list constaing cloumns to check for completeness
 Returns:
     True if the fill rate of at least one specified combination is greater than 80%, False otherwise.
 Raises:
     ValueError: If the dataframe is empty or if any of the required columns are missing.
 """
 if df.empty:
   raise ValueError("DataFrame is empty.")
  
 is_column_exist = False
 
 for columns in columns_to_check:  
   
  if isinstance(columns, list):     
    final_result += f"Mandatory Attributes : {columns} " + "\n"      
    missing_columns = [col for col in columns if col not in df.columns]

    if missing_columns: 
      final_result +=  f"Column does not exist : {missing_columns} "+   "\n"
      found_columns = [col for col in columns if col in df.columns]
      final_result +=  f"Column found : {found_columns} "+   "\n"

    else:  
       final_result +=  f"Fill rate threshold :  {fill_rate_threshold * 100}% "+   "\n"
       dict_fillrate = {}
       for column in columns:
          fill_rate = calculate_fill_rate(df, column)
          dict_fillrate[column] = fill_rate
       
       for key, value in dict_fillrate.items():
        final_result +=  f"Fill rate of column {key} : {value * 100}%"+   "\n"
  
       if all(fill_rate >= fill_rate_threshold for fill_rate in dict_fillrate.values()):      
        is_column_exist = True        
       else:
          # Identify columns with fill rates below the threshold
          columns_below_threshold = [column for column, fill_rate in dict_fillrate.items() if fill_rate < fill_rate_threshold]         
          final_result += f"Fill rate is less than {fill_rate_threshold * 100}% for column {columns_below_threshold}"+ "\n"                 
                      
  elif isinstance(columns, str):    
    final_result += f"Mandatory Attributes : {columns} " + "\n"  
    if columns not in df.columns:         
        final_result +=  f"Column does not exist: {columns}"+   "\n"   
    else:      
      fill_rate = calculate_fill_rate(df, columns)
      final_result +=  f"Fill rate of column {columns} : {fill_rate}% "+   "\n"
      if fill_rate >= fill_rate_threshold:
        is_column_exist = True      
      else:        
        final_result +=  f"Fill rate is less than {fill_rate_threshold * 100}% for {columns} "+   "\n"   
      
 return is_column_exist, final_result 

def validate_encoding(file_path, encoding):
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            file.read()
        return True
    except (UnicodeDecodeError, LookupError) as e:         
        return False

def check_alphabets_and_trim_spaces(df, column_name,referene_column,final_result=''):        
    if column_name in df.columns:
      non_alphabetical_values = df[~df[column_name].astype(str).str.replace(' ', '').str.isalpha()]     
      if not non_alphabetical_values.empty: 
          for index,row in non_alphabetical_values.iterrows():    
            final_result += f"Non alphabetic characters found {row[column_name]} for {row[referene_column]} " + "\n"
            is_data_remidated = True
          final_result += f"Removing Non-alphabetical characters" + "\n"         
          df[column_name] = df[column_name].str.replace(r'[^a-zA-Z]', '', regex=True)
          return True,final_result, df ,is_data_remidated   
      else:
          final_result += f"No non-alphabetical values found in column {column_name}:" + "\n"   
          is_data_remidated = False
          return True, final_result,df,is_data_remidated
    else:
        final_result += f"Column '{column_name}' not found in DataFrame." + "\n"
        return False,final_result,df,False
 
def validate_email(df, column_name,reference_column,final_result=''):    
    pattern = r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[a-zA-Z]{2,})+$'
    if column_name not in df.columns:
        final_result += f"Column '{column_name}' not found in DataFrame." + "\n"
        return False, final_result
    # invalid_emails = df[~df[column_name].astype(str).str.replace(' ', '').str.match(pattern)] 
    invalid_emails = df[~df[column_name].astype(str).str.match(pattern)]      
    if not invalid_emails.empty:       
       final_result += f"Invalid email id found in column {column_name}" + "\n"   
       for index, row in invalid_emails.iterrows():     
        final_result += f"Invalid email {row[column_name]} for {reference_column} {row[reference_column]} "+ "\n"       
       return False, final_result
    else:
        return True, final_result

def validate_phonenumbers(df, phonenumbers_column, referencecolumn,final_result = '', region='US'):
  """
  Validates a phone number and reports if it has extra spaces.
  Args:
      phone_number (str): The phone number string.

  Returns:
      None: Prints a message if the number is invalid due to extra spaces.
  """
  if phonenumbers_column not in df.columns:    
    final_result += f"Column '{phonenumbers_column}' does not exist in the DataFrame."
    return False, final_result

  # Remove leading and trailing spaces
  df[phonenumbers_column] = df[phonenumbers_column].str.strip()

  # valid_chars = "0123456789()+-"
  # df[phonenumbers_column] = df[phonenumbers_column].apply(lambda x: "".join([c for c in x if c in valid_chars]))
  # print(df[phonenumbers_column])

  is_invalid_phone= True
  
  # for phone_number in df[phonenumbers_column]: 
  for phone_number, individual_id in zip(df[phonenumbers_column], df[referencecolumn]): 
    if (phone_number == 'NaN' ):            
      if type(phone_number) == float or type(phone_number) == str:
        final_result += f"Phone number is empty: {phone_number} for cid {individual_id}"+ "\n"        
        continue
    elif pd.isna(phone_number):  
      final_result += f"Phone number is empty : {phone_number} for cid {individual_id}"+ "\n"      
      continue
    else:
      try:
        parsed_number = phonenumbers.parse(phone_number,region)           
        valid = phonenumbers.is_valid_number(parsed_number)
        if valid:
          try:    
            number_length = 0        
            number_length = len(phone_number.replace(" ", "").replace("-", "").replace("(", "").replace(")", ""))          
          except TypeError as e:                     
            final_result += f"Exception while checking length:  {phone_number} Exception {e} for cid {individual_id}"  + "\n"
          
          if number_length < 9 or number_length > 15:
            final_result += f"Invalid phone number: {phone_number} for cid: {individual_id}. Length of Phonenumber is {number_length}, length is out of range." + "\n"
            is_invalid_phone = False
                
        else:          
          final_result += f"Invalid phone number: {phone_number} for cid {individual_id}" + "\n"
          is_invalid_phone = False

      except phonenumbers.phonenumberutil.NumberParseException:      
        final_result += f"Invalid phone number: {phone_number}. phonenumbers parsing error for cid {individual_id}" + "\n"
        is_invalid_phone = False
          
  if is_invalid_phone == False:
    return False, final_result   
  else:
     return True, final_result    
  


def is_number(value):
  try:
      float(value)
      return True
  except ValueError:
      return False
    
def validate_date(df,column_name,data_format,referencecolumn,final_result=''):
    is_valid_date = True    
    
    if column_name not in df.columns.to_list():        
        final_result += f"Column '{column_name}' not found in DataFrame."
        return False, final_result
    else:  
      for date_string, individual_id in zip(df[column_name], df[referencecolumn]):        
      # for date_string in df[column_name]:
          # Attempt to parse the date string
          try:
              if type(date_string) == float:
                 print(f"Date is type float: {date_string}")
                 date_string = str(date_string)
              if date_string != 'NaN':   
                datetime.strptime(date_string, data_format)  
              # else:
                # final_result += f"Date is empty: {date_string}"+ "\n"
                # is_valid_date = False              
          except ValueError:
              # Raised when the format is incorrect
              final_result += f"Invalid date format for cid {individual_id}: {date_string}"+ "\n"
              is_valid_date = False
      return is_valid_date, final_result  
 
def validate_number_columns(df,columns_list,final_result=''):   
    is_valid_number = True
    for column in columns_list:
      if column not in df.columns:
        final_result += f"Column '{column}' not found in DataFrame."
        return False, final_result
      else:
        for value in df[column]:
          if not is_number(value):
              final_result += f"Invalid number: {value} in column {column}"+ "\n"
              is_valid_number = False
    return is_valid_number, final_result

# Function to check if timestamp is in the correct format
def check_timestamp_format(dataframe, timestamp_column,referencecolumn,timestamp_format = '%Y-%m-%d %H:%M:%S.%f', given_timezone = '',final_result=''):
    if timestamp_column not in dataframe.columns:
        final_result += f"Column '{timestamp_column}' not found in DataFrame."
        return False, final_result
    # Convert string to datetime object
    is_timestamp_and_timezone_match = True
    for row, individual_id in zip(dataframe[timestamp_column], dataframe[referencecolumn]):    
    # for index, row in dataframe.iterrows():
        try:     
            dt = datetime.strptime(str(row[timestamp_column]),timestamp_format )
            # Get timezone information
            if given_timezone != '':
                tz = pytz.timezone(pytz.country_timezones('US')[0])
                dt_tz = tz.localize(dt)                
                # Compare timezones
                if dt_tz.tzinfo == pytz.timezone(given_timezone):                  
                    return True
                else:                    
                    is_timestamp_and_timezone_match = False
                    final_result += f"Invalid timestamp format for cid {individual_id}: {row[timestamp_column]} Timezone not matched" + "\n"
        except ValueError:
            final_result += f"Invalid timestamp format for cid {individual_id}: {row[timestamp_column]}" + "\n"
            is_timestamp_and_timezone_match = False   
    return is_timestamp_and_timezone_match,final_result         

def detect_and_convert_timezones(df, timestamp_column, target_format="%Y-%m-%dT%H:%M:%S", target_timezone="UTC",referencecolumn = 'IndividualID',final_result=''):
    """
    Detects the timezone of timestamps in a DataFrame, checks format,
    and converts to the specified format and timezone.

    Args:
        df (pd.DataFrame): The DataFrame containing the timestamp column.
        timestamp_column (str): The name of the timestamp column.
        target_format (str, optional): The desired output format for timestamps.
            Defaults to "%Y-%m-%dT%H:%M:%S".
        target_timezone (str, optional): The target timezone to convert to.
            Defaults to "UTC".
    Returns:        
    """
    is_timestamp_and_timezone_match = True
    
    if timestamp_column not in df.columns:
        final_result += f"Column '{timestamp_column}' not found in DataFrame." + "\n"
        return final_result,False
    # for timestamp_str, individual_id in zip(df[timestamp_column], df[referencecolumn]):  
    for index, row in df.iterrows():  
        try:    
            timestamp_str = row[timestamp_column]            
            individual_id = row[referencecolumn] 
            if not pd.isna(timestamp_str):               
              if timestamp_str == 'NaN' or timestamp_str == 'NAN' or timestamp_column == 'nan'or timestamp_str == float('nan') or timestamp_str =='':
                final_result += f"Empty timestamp: {timestamp_str} for cid: {individual_id}" +"\n"
              elif timestamp_str.startswith("-"):
                final_result += f"Invalid timestamp: {timestamp_str} for cid: {individual_id}" +"\n" 
                is_timestamp_and_timezone_match = False
              else:
                length_timetamp = len(timestamp_str)
                # Check if there's more than one element (indicating double timestamps)
                if length_timetamp < 35:
                  # Attempt to parse timestamp with or without explicit timezone information
                  if " " in timestamp_str.strip():  # Timestamp with space separator (likely offset)  
                      timestamp_values = timestamp_str.split(" ")                
                      if 'T' in timestamp_str and len(timestamp_values) == 2:                    
                          timestamp_value_withouttimezone = timestamp_values[0]                   
                          datetime_obj = pd.to_datetime(timestamp_value_withouttimezone.split("T")[0] + " " + timestamp_value_withouttimezone.split("T")[1])
                          timezone = pytz.timezone(timestamp_values[1])
                          datetime_obj = timezone.localize(datetime_obj)                          
                      else: 
                          datetime_obj = pd.to_datetime(timestamp_values[0] + " " + timestamp_values[1])   
                                     
                  elif 'T' in timestamp_str:  # Timestamp with 'T' separator and optional timezone offset  
                      datetime_obj = pd.to_datetime(timestamp_str.split("T")[0] + " " + timestamp_str.split("T")[1])
                  else:                          
                       datetime_obj = None                        

                  final_result += f'datetime_obj: {datetime_obj}' + "\n"    
                  final_result += f'tzinfo {datetime_obj.tzinfo}'+ "\n"
              
                  # Check if timezone information is explicitly provided
                  if datetime_obj.tzinfo != None:                
                      # Convert to target timezone if necessary (assuming datetime_obj is timezone-aware)
                      if datetime_obj.tzinfo != pytz.timezone(target_timezone):  
                          final_result += f'Converting to target timezone: {target_timezone}' + "\n"                  
                          datetime_obj = datetime_obj.astimezone(pytz.timezone(target_timezone))                    
                  else:                
                      # Localize naive datetime objects to the given target timezone  
                      final_result += f'Localizing to target timezone: {target_timezone}' + "\n"          
                      datetime_obj = pytz.timezone(target_timezone).localize(datetime_obj)
                      
                  # Format the converted datetime object
                  final_result += f'converted value {datetime_obj.strftime(target_format)}' + "\n" 
                  df.loc[df[timestamp_column] == timestamp_str, timestamp_column] = datetime_obj.strftime(target_format)
                else:
                  final_result += f"Invalid timestamp: {timestamp_str} for cid: {individual_id}" +"\n" 
                  is_timestamp_and_timezone_match = False  
            else:
              final_result += f"Empty timestamp: {timestamp_str} for cid: {individual_id}" +"\n"
        except:
            is_timestamp_and_timezone_match = False
            final_result += f"Invalid timestamp: {timestamp_str} for cid: {individual_id}" +"\n"                      
            
    return final_result,is_timestamp_and_timezone_match

def upload_to_s3(local_file_path, bucket_name, s3_key,final_result =''):
    # Create an S3 client
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )
    current_date = datetime.now()
    formatted_date = current_date.strftime('%Y%m%d')
    dirPath = s3_key.rsplit('/', 1)[0]    
    fileName = s3_key.rsplit('/', 1)[1]    
    newKey = dirPath + '/' + formatted_date + '/' + fileName
    # Upload file to the specified S3 bucket
    try:
        s3.upload_file(local_file_path, bucket_name, newKey)
        final_result+= f"File uploaded successfully to s3://{s3_key}" + "\n"
    except Exception as e:
        final_result+= f"Error uploading file: {e}" + "\n"
    return final_result    

def validate_address(county, zip_code, state):
    address = f"{county}, {state} {zip_code}"
    try:
        parsed_address, address_type = usaddress.tag(address)        
        if address_type != 'Ambiguous':            
            return True
    except usaddress.RepeatedLabelError:
        pass
    return False       

def move_s3_objects(key: str) -> None:
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )
    current_date = datetime.now()
    formatted_date = current_date.strftime('%Y%m%d')
    dirPath = key.rsplit('/', 1)[0]
    fileName = key.rsplit('/', 1)[1]
    newKey = dirPath + '/' + formatted_date + '/' + fileName
    if s3DictionaryPrefix in key:
        destination_key = newKey.replace(s3DictionaryPrefix, s3DictionaryFailedPath, 1)
    if s3RawFilesPrefix in key:
        destination_key = newKey.replace(s3RawFilesPrefix, s3RawFilesFailed, 1)
    # Copy the object to the new location
    print(f"copying file from {key} to failed location of S3  {destination_key}")
    s3.copy_object(
            Bucket=BUCKET_NAME,
            CopySource={'Bucket': BUCKET_NAME, 'Key': key},
            Key=destination_key
        )
    print(f"Deleting file from raw location of S3:  {key}")
    s3.delete_object(Bucket=BUCKET_NAME, Key=key)

def dataQualityChecksProcess():  
    config = read_config(dataQualityJsonFile)
    # config = read_config('config.json')

    final_validation_result = True
    final_result = ''
    failed_file_list = []
    encoding = ''

    for file_data in config["Files"]: 
        isfile_remediated = False   
        category = file_data["DataQualityChecks"]["Category"]
        completeness = category.get("Completeness", [])    
        conformity = category.get("Conformity", [])
        fileformatConformity = category.get("File-Format Conformity", [])
        uniqueness = category.get("Uniqueness", [])
        consistency = category.get("Consistency", [])
        file_name = file_data['Filename']
        Curated_file = file_data['CuratedFileName']
        # print(f'file_name {file_name}')
        # print(f'filename {file_name}')
        # print(f'consistency {consistency}, type {type(consistency)}')
        # print(f'uniqueness {uniqueness}, type {type(uniqueness)}')
        # print(f'completeness {completeness}, type {type(completeness)}')
                        
        final_result += f"Validation Report for File: {file_name} " + "\n"
        final_result += "********************************************" + "\n"
    
        if fileformatConformity['Rule']['Enable']:           
            if fileformatConformity['Description'] == 'File-Format Conformity':
                encoding = fileformatConformity['Rule']['Encoding']  
                rule_id = fileformatConformity['Id']
                rule_name = fileformatConformity['Description']              
                final_result += f"Rule Id : {rule_id} " + "\n"
                final_result += f"Rule Name : {rule_name} " + "\n"
                final_result += f"Required encoding : {encoding} " + "\n"
                            
                if validate_encoding(file_name, encoding):            
                    final_result += f"{rule_name } Status: Passed " + "\n"
                    final_result += "********************************************" + "\n"  
                else:
                    final_validation_result = False
                    failed_file_list.append(file_name)
                    final_result += f"{rule_name} Status: Failed " + "\n"    
                    final_result += "********************************************" + "\n"   
                    continue  

        else :
            final_result += f"{completeness['Description']} Status: Disabled " + "\n"  
            final_result += "********************************************" + "\n"                                
        
        df_data = read_data("", file_name,encoding) 
        if df_data is None:
            failed_file_list.append(file_name)
            final_validation_result = False
            final_result += f"File {file_name} is blank. " + "\n"  
            continue
                        
        if completeness != {}:
            rule_id = completeness['Id']
            rule_name = completeness['Description']        
            fillrate_value = completeness['Rule']['FillRate']            
            final_result += f"Rule Id : {rule_id} " + "\n"
            final_result += f"Rule Name : {rule_name} " + "\n"
            
        # Process Completeness checks
            if completeness['Rule']['Enable']:                
                columns_to_check_completeness = completeness['Rule']['ColumnName']                    
                is_exists, final_result= completeness_check(df_data,columns_to_check_completeness,fillrate_value,final_result)
                if is_exists:
                    final_result += f"{rule_name} Status: Passed " + "\n"
                else:
                    final_validation_result = False
                    failed_file_list.append(file_name)
                    final_result += f"{rule_name} Status: Failed " + "\n"                 
                final_result += "********************************************" + "\n" 
            else :
                final_result += f"{completeness['Description']} Status: Disabled " + "\n"  
                final_result += "********************************************" + "\n"           

    # # Process Conformity checks
        if conformity != []:
            for conformity_check in conformity:               
                rule_id = conformity_check['Id']
                rule_name = conformity_check['Description']         

                if rule_name == 'First Name and Last Name check':  
                    final_result += f"Rule Id : {rule_id} " + "\n"
                    final_result += f"Rule Name : {rule_name} " + "\n"                          
                    
                    if conformity_check['Rule']['Enable']: 
                        columns = conformity_check['Rule']['ColumnName']
                        reference_column = conformity_check['Rule']['ReferenceColumnName']
                        for str_column in columns: 
                            try:
                                                              
                                if df_data[str_column][df_data[str_column].notnull()].empty:
                                    final_result += f"{str_column} : is empty" + "\n"
                                    final_result += f"{rule_name} Status: Passed  for column {str_column} " + "\n"
                                else:
                                    df_data[str_column] = df_data[str_column][df_data[str_column].notnull()].str.strip()                                    
                                    isalphabetical, final_result, df_data,isfile_remediated = check_alphabets_and_trim_spaces(df_data,str_column,reference_column,final_result) 
                                    if isalphabetical:                         
                                        final_result += f"{rule_name} Status: Passed  for column {str_column} " + "\n"
                                    else:                                
                                        final_result += f"{rule_name} Status: Failed for column {str_column}" + "\n"    
                            except Exception as e:
                                    final_validation_result = False
                                    final_result += f"{rule_name} Status: Failed with error {e}" + "\n" 
                                    
                        final_result += "********************************************" + "\n"   
                    else:
                        final_result += f"{rule_name} Status: Disabled  for {file_name}" + "\n"            
                        final_result += "********************************************" + "\n"  
                      
                elif rule_name== "Email Id check":                
                    final_result += f"Rule Id : {rule_id} " + "\n"
                    final_result += f"Rule Name : {rule_name} " + "\n" 

                    if conformity_check['Rule']['Enable']: 
                        email_column = conformity_check['Rule']['ColumnName']
                        reference_column = conformity_check['Rule']['ReferenceColumnName']
                        is_invalid_email, final_result = validate_email(df_data,email_column,reference_column,final_result)
                        if is_invalid_email:
                            final_result += f"{rule_name} Status: Passed " + "\n"
                        else:
                            final_validation_result = False
                            failed_file_list.append(file_name)
                            final_result += f"{rule_name} Status: Failed " + "\n"   
                            
                        final_result += "********************************************" + "\n"
                    else:
                        final_result += f"{rule_name} Status: Disabled  for {file_name}" + "\n"            
                        final_result += "********************************************" + "\n"     
                    
                elif rule_name== "Phone number check":                
                    final_result += f"Rule Id : {rule_id} " + "\n"
                    final_result += f"Rule Name : {rule_name} " + "\n" 
                    
                    if conformity_check['Rule']['Enable']:                     
                        phonenumbers_column = conformity_check['Rule']['ColumnName']
                        region = conformity_check['Rule']['Region']
                        reference_column = conformity_check['Rule']['ReferenceColumnName']
                        if phonenumbers_column in df_data.columns:                            
                            df_data[phonenumbers_column] = df_data[phonenumbers_column].fillna('NaN')
                            is_valid_phone, final_result = validate_phonenumbers(df_data,phonenumbers_column,reference_column,final_result,region)
                            if is_valid_phone:
                                final_result += f"{rule_name} Status: Passed " + "\n"
                            else:
                                final_validation_result = False
                                failed_file_list.append(file_name)
                                final_result += f"{rule_name} Status: Failed " + "\n"                           
                            final_result += "********************************************" + "\n"
                        else:
                            final_validation_result = False
                            failed_file_list.append(file_name)                                                    
                            final_result += f"Column '{phonenumbers_column}' not found in DataFrame." + "\n"
                            final_result += f"{rule_name} Status: Failed " + "\n"   
                            final_result += "********************************************" + "\n"    
                    else:
                        final_result += f"{rule_name} Status: Disabled  for {file_name}" + "\n"            
                        final_result += "********************************************" + "\n"

                elif rule_name== "Date Format check":                
                    final_result += f"Rule Id : {rule_id} " + "\n"
                    final_result += f"Rule Name : {rule_name} " + "\n" 
                    
                    if conformity_check['Rule']['Enable']: 
                        date_column = conformity_check['Rule']['ColumnName']
                        Format = conformity_check['Rule']['Format']
                        if date_column in df_data.columns:
                            df_data[date_column] = df_data[date_column].fillna('NaN')
                            reference_column = conformity_check['Rule']['ReferenceColumnName']
                            is_valid_date, final_result = validate_date(df_data,date_column,Format,reference_column,final_result)
                            if is_valid_date:
                                final_result += f"{rule_name} Status: Passed " + "\n"
                            else:
                                final_validation_result = False
                                failed_file_list.append(file_name)
                                final_result += f"{rule_name} Status: Failed " + "\n"                           
                            final_result += "********************************************" + "\n"
                        else:
                            final_result += f"{rule_name} Status: Disabled  for {file_name}" + "\n"            
                            final_result += "********************************************" + "\n"        
        
                elif rule_name== "Number Column check":  
                    final_result += f"Rule Id : {rule_id} " + "\n"
                    final_result += f"Rule Name : {rule_name} " + "\n" 
                    
                    if conformity_check['Rule']['Enable']: 
                        number_columns = conformity_check['Rule']['ColumnName']
                    
                        is_valid_date, final_result = validate_number_columns(df_data,number_columns,final_result)
                        if is_valid_date:
                            final_result += f" {rule_name} Status: Passed " + "\n"
                        else:
                            final_validation_result = False
                            failed_file_list.append(file_name)
                            final_result += f"{rule_name} Status: Failed " + "\n"                           
                        final_result += "********************************************" + "\n"
                    else:
                        final_result += f"{rule_name} Status: Disabled  for {file_name}" + "\n"            
                        final_result += "********************************************" + "\n" 

                elif rule_name== "Timestamp check":  
                    final_result += f"Rule Id : {rule_id} " + "\n"
                    final_result += f"Rule Name : {rule_name} " + "\n" 
                    
                    if conformity_check['Rule']['Enable']: 
                        timestamp_column = conformity_check['Rule']['ColumnName']
                        timestamp_format = conformity_check['Rule']['TimestampFormat']
                        reference_column = conformity_check['Rule']['ReferenceColumnName']
                        target_timezone = conformity_check['Rule']['TargetTimezone']
                        if timestamp_column in df_data.columns:
                            # df_data[timestamp_column] = df_data[timestamp_column].fillna('NaN')
                        
                            final_result,is_timestamp_and_timezone_match = detect_and_convert_timezones(df_data,timestamp_column,timestamp_format,target_timezone,reference_column,final_result)
                            
                            if is_timestamp_and_timezone_match:
                                final_result += f"{rule_name} Status: Passed " + "\n"
                                isfile_remediated = True
                            else:
                                final_validation_result = False
                                failed_file_list.append(file_name)
                                final_result += f"{rule_name} Status: Failed " + "\n"                           
                            final_result += "********************************************" + "\n"
                        else:
                            final_result += f"Column '{timestamp_column}' not found in DataFrame." + "\n"
                            failed_file_list.append(file_name)
                            final_result += "********************************************" + "\n"    
                    else:
                        final_result += f"{rule_name} Status: Disabled  for {file_name}" + "\n"            
                        final_result += "********************************************" + "\n"  
        
                elif rule_name== "Address check":  
                    final_result += f"Rule Id : {rule_id} " + "\n"
                    final_result += f"Rule Name : {rule_name} " + "\n" 
                    
                    if conformity_check['Rule']['Enable']: 
                        columnnames = conformity_check['Rule']['ColumnName'] 
                        print(columnnames)
                          
                        if set(columnnames).issubset(set(df_data.columns)):                     
                            reference_column = conformity_check['Rule']['ReferenceColumnName']                                            
                            # Add a new column 'Address_Valid' to store validation result
                            df_data['Address_Valid'] = df_data.apply(lambda row: validate_address(row[columnnames[0]], row[columnnames[1]], row[columnnames[2]]), axis=1)
                            invalid_address_df = df_data[df_data['Address_Valid'] == False]
                            if invalid_address_df.empty:
                                final_result += f" {rule_name} Status: Passed " + "\n"                            
                            else:
                                final_validation_result = False
                                for index, row in invalid_address_df.iterrows():
                                    final_result += f"Invalid address: 'county:'{row[columnnames[0]]},'State'{row[columnnames[1]]}, 'Zip' {row[columnnames[2]]} for cid {row[reference_column]}" + "\n"
                                final_result += f"{rule_name} Status: Failed " + "\n"  
                                failed_file_list.append(file_name)                         
                            final_result += "********************************************" + "\n"
                        else:
                            missing_columns = set(columnnames) - set(df_data.columns)
                            final_result += f"Column '{missing_columns}' not found in DataFrame." + "\n"  
                            final_result += f"{rule_name} Status: Failed " + "\n"
                            failed_file_list.append(file_name)                              
                    else:
                        final_result += f"{rule_name} Status: Disabled  for {file_name}" + "\n"            
                        final_result += "********************************************" + "\n" 
                        
        if uniqueness!= {}:
            if uniqueness['Rule']['Enable']:                   
                rule_id = uniqueness['Id']
                rule_name = uniqueness['Description']               
                         
                if rule_name == "Uniqueness Check":                    
                    final_result += f"Rule Id : {rule_id} " + "\n"
                    final_result += f"Rule Name : {rule_name} " + "\n"               
                    duplicate_rows = df_data[df_data.duplicated()]               
                   
                    if duplicate_rows.empty:                        
                        final_result += f" {rule_name} Status: Passed " + "\n"                  
                    else:     
                        final_result += f"{len(duplicate_rows)} duplicate rows found in data. Removing duplicate rows." + "\n"    
                        df_data.drop_duplicates(inplace=True)    
                        isfile_remediated = True         
                        
                        final_result += f"{rule_name} Status: Passed " + "\n"                           
                    final_result += "********************************************" + "\n"
            else:
                final_result += f"{rule_name} Status: Disabled  for {file_name}" + "\n"            
                final_result += "********************************************" + "\n"                                      

        if consistency != {}:            
            if consistency['Rule']['Enable']: 
                rule_id = consistency['Id']
                rule_name = consistency['Description']               
                      
                if rule_name == "Null value check":                                      
                    final_result += f"Rule Id : {rule_id} " + "\n"
                    final_result += f"Rule Name : {rule_name} " + "\n"               
                    
                    has_null = df_data.isnull().any().any()
                    has_unknown = df_data.eq('Unknown').any().any()
                    has_na = df_data.eq('NA').any().any() 
                    # print(f'has_null {has_null} has_unknown {has_unknown} has_na {has_na}')             
                    
                    if has_null == False and has_unknown == False and has_na == False:                        
                        final_result += f" {rule_name} Status: Passed " + "\n"                  
                    else:      
                        final_result += "Inconsistant null value found in data. Values being updated to consistant value" + "\n"                  
                          # Replace specific strings with np.nan
                        ConsistentValue = consistency['Rule']['ConsistentValue']
                        df_data.replace(['null', 'Unknown', 'NA','NaN','Nan',np.nan],ConsistentValue, inplace=True)
                        isfile_remediated = True
                                                              
                        final_result += f"{rule_name} Status: Passed. Consistant null value updated in data " + "\n"                           
                    final_result += "********************************************" + "\n"
            else:
                final_result += f"{rule_name} Status: Disabled  for {file_name}" + "\n"            
                final_result += "********************************************" + "\n"
              
        if isfile_remediated: 
            df_data.to_csv(file_name, index=False)           
            final_result = upload_to_s3(file_name, BUCKET_NAME, Curated_file,final_result) 
        
    print(final_result)          

    if not final_validation_result:
        for file_name in failed_file_list:
              '''code to move file to failure location in S3 and to delete downloaded file from container'''
              print(f"removing file from container: {file_name}")
              os.remove(file_name)
              if locatPathToDownloadRawFiles in file_name:
                    s3KeyToDelete = file_name.replace(locatPathToDownloadRawFiles, s3RawFilesPrefix, 1)
              if locatPathToDownloadDictionaryFiles in file_name:
                    s3KeyToDelete = file_name.replace(locatPathToDownloadDictionaryFiles, s3DictionaryPrefix, 1)
                
              move_s3_objects(s3KeyToDelete)
        raise ValueError("Data quality checks failed. Please see the logs for more details.")      


        
