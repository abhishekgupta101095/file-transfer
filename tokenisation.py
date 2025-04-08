import pandas as pd
import scripts.config as config
import hmac
import hashlib
import requests

token = config.token
tokenization_required= config.tokenization_required
files_with_columns_to_tokenize = config.files_with_columns_to_tokenize
url = config.tokenization_url

def get_key(token):   
    headers = {
        "X-Vault-Token": token
    }
    response = requests.get(url, headers=headers, verify=False)
    if response.status_code == 200:
        return response.json().get('data').get('value')
    else:
        raise ValueError(f"Error in getting key")


def generate_token(x,key):
    hmac_digest = hmac.new(key.encode("utf-8"), x.encode("utf-8"), hashlib.sha256).hexdigest()
    return hmac_digest



def tokenizationProcess():
    if tokenization_required:
        key = get_key(token)
        for filename,columns_to_tokenize in files_with_columns_to_tokenize.items():
            df = pd.read_csv(filename)
            columns_present = df.columns
            for col in columns_to_tokenize:
                if col in columns_present:
                    df[col] = df[col].apply(lambda x: generate_token(x, key))
                else:
                    raise ValueError(f"This column in DF is not present: {col} in file : {filename}")
                    
            df.to_csv(filename,index=False)