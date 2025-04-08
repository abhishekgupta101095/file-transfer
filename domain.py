# Copyright © 2024 Accenture. All rights reserved.

# File: domain.py
# Author: abhishek.cw.gupta
# Date: 28-May-2024
# Description: Create domain table to map domains with uuid.

# import pandas as pd
# import uuid
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

domains_table_name = config.domains_table_name

connection_string = f'postgresql://{postgresUser}:{postgresPassword}@{postgresHost}:{postgresPort}/{postgresDB}'

def create_domains():

    data = [['Media/Marketing', 1], ['Content', 1], ['Inteligence', 1]]
    domains = pd.DataFrame(data, columns=['domain_name', 'domain_id'])
    domains['domain_id'] = [uuid.uuid4() for _ in range(len(domains.index))]
    Helper.write_dataframe_to_postgresql_in_chunks(domains, domains_table_name, chunksize=10000, connection_string=connection_string)

if __name__ == '__main__':
   create_domains()
