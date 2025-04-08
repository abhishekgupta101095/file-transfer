import sqlalchemy
from sqlalchemy import create_engine
import pandas as pd

def write_dataframe_to_postgresql_in_chunks(df, table_name, chunksize=10000, connection_string=None):
    """Writes a DataFrame to a PostgreSQL database in chunks.
 
    Args:
        df (pd.DataFrame): The DataFrame to write.
        table_name (str): The name of the target table.
        chunksize (int, optional): The number of rows to write in each chunk. Defaults to 10000.
        connection_string (str, optional): The connection string to the database.
            If not provided, uses environment variables.
    """
 
    if connection_string is None:
        connection_string = "postgresql://user:password@host:port/database"  # Replace with your credentials
 
    db = create_engine(connection_string)
    print(connection_string)
    conn = db.connect()
    df.to_sql(table_name,con= conn,schema='public',if_exists='append',index=False)
    print("Data inserted successfully") 

def select_table_data(tableName,cols=[],connection_string=None):
    try:
        """ This method select data from table and returns table data as dataframe"""
        if connection_string is None:
            connection_string = "postgresql://user:password@host:port/database"  # Replace with your credentials
 
        db = create_engine(connection_string)
        if len(cols) == 0:
            select_cols = '*'
        else:
            select_cols = ','.join(f'"{column}"' for column in cols)

        conn = db.connect()  
        select_command = f""" SELECT {select_cols} FROM "{tableName}" """
                   
        sql_df = pd.DataFrame()
        sql_df = pd.read_sql(select_command,conn)      
        conn.close()  
        return sql_df 
    except Exception as e:
        db.dispose()
        conn.close()
        print(f"Exception from select_table_data method: {e} ",'error')
        raise e    
