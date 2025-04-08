# Copyright © 2024 Accenture. All rights reserved.

# File: DataStalenessOutliersChecks.py
# Author: chitresh.goyal
# Date: 9-April-2024
# Description: Data Integrity Check

# -*- coding: utf-8 -*-

import pandas as pd
import json
import os

def read_config(config_file_path):
    try:
        with open(config_file_path, 'r') as config_file:
            config = json.load(config_file)
        return config
    except Exception as e:
        print(f"Error reading config JSON: {str(e)}")

# Load configuration
config = read_config('config.json')
file_config = config['Files'][6]
integrity_check_config = file_config['DataIntegrityCheck']

file1=integrity_check_config['File1']
file2=integrity_check_config['File2']
# Load the two files into DataFrames
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)


# Extract the Individual_ID columns from both DataFrames
ids_df1 = set(df1[integrity_check_config['KeyColumn']].unique())
ids_df2 = set(df2[integrity_check_config['KeyColumn']].unique())

# Find the common and missing Individual_IDs
common_ids = ids_df1.intersection(ids_df2)
missing_in_df2 = ids_df1.difference(ids_df2)
missing_in_df1 = ids_df2.difference(ids_df1)

# Print the results
file1_name = os.path.basename(file1)
file2_name = os.path.basename(file2)

print(f"Number of common Individual_IDs: {len(common_ids)}")
print(f"Number of Individual_IDs in {file1_name} but not in {file2_name}: {len(missing_in_df2)}")
print(f"{missing_in_df2}")
print(f"Number of Individual_IDs in {file2_name} but not in {file1_name}: {len(missing_in_df1)}")
print(f"{missing_in_df1}")

# Optional: Save the results to a file
with open(integrity_check_config['OutputFile'], 'w') as f:
    f.write(f"Number of common Individual_IDs: {len(common_ids)}\n")
    f.write(f"Number of Individual_IDs in {file1_name} but not in {file2_name}: {len(missing_in_df2)}\n")
    f.write(f"{missing_in_df2}\n")
    f.write(f"Number of Individual_IDs in {file2_name} but not in {file1_name}: {len(missing_in_df1)}\n")
    f.write(f"{missing_in_df1}\n")
    

