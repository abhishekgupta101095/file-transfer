# Copyright © 2024 Accenture. All rights reserved.

# File: DataStalenessOutliersChecks.py
# Author: chitresh.goyal
# Date: 4-April-2024
# Description: Data Staleness and Data Outliers Checks

# -*- coding: utf-8 -*-

import pandas as pd
import json
from datetime import datetime

def read_config(config_file_path):
    try:
        with open(config_file_path, 'r') as config_file:
            config = json.load(config_file)
        return config
    except Exception as e:
        print(f"Error reading config JSON: {str(e)}")

def calculate_age(birth_date, date_format):
    if pd.isnull(birth_date):
        return None
    today = datetime.today()
    birth_date = datetime.strptime(birth_date, date_format)
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return int(age)

def check_age_group(age, age_group, max_age):
    if age is None:
        return False
    if '+' in age_group:
        min_age = int(age_group.replace('+', ''))
        result = min_age <= age <= max_age
    elif '-' in age_group:
        min_age, max_age = map(int, age_group.split('-'))
        result = min_age <= age <= max_age
    else:
        result = False
    print(f"Checking age {age} against age group {age_group} | Result is {result}")
    return result

# Load configuration
config = read_config('config.json')
file_config = config['Files'][5]
DataOutlierCheck = file_config['DataOutlierCheck']

# Load CSV data
df = pd.read_csv(file_config['Filename'])

# Initialize the outlier_column variable
outlier_column = None

print(DataOutlierCheck.get('AgeCalculation', {}).get('Rule'))

if DataOutlierCheck.get('AgeCalculation', {}).get('Rule').get('Enable', False):
    print("Age calculation enabled")
    age_calc_rule = DataOutlierCheck.get('AgeCalculation',{}).get('Rule')
    birth_date_column = age_calc_rule['BirthDateColumn']
    age_column = age_calc_rule['AgeColumn']
    date_format = age_calc_rule['Format']
    df[age_column] = df[birth_date_column].apply(lambda x: calculate_age(x, date_format))
    print(f"Age column '{age_column}' added to DataFrame")

if DataOutlierCheck.get('AgeOutlierDetection', {}).get('Rule').get('Enable', False):
    print("Age outlier detection enabled")
    age_outlier_rule = DataOutlierCheck.get('AgeOutlierDetection', {}).get('Rule')
                                                                           
    age_group_column = age_outlier_rule['AgeGroupColumn']
    outlier_column = age_outlier_rule['OutlierColumn']
    max_age = age_outlier_rule['MaxAge']
    df[outlier_column] = df.apply(lambda row: not check_age_group(row[age_column], row[age_group_column], max_age), axis=1)
    print(f"Outlier column '{outlier_column}' added to DataFrame")

# Verify the DataFrame has the new columns
print("Verifying the new columns are in DataFrame:")
print(df.head())

# Output the rows with outlier ages only if outlier_column is defined
if outlier_column:
    outliers = df[df[outlier_column]]
    print("Outliers detected:")
    print(outliers)

    df[outlier_column] = df[outlier_column].map({True: 'Invalid', False: 'Valid'})

# Append Age Outlier in csv file
output_path = file_config['CuratedFileName']
df.to_csv(output_path, index=False)
print(f"CSV file with new columns has been saved to {output_path}")



