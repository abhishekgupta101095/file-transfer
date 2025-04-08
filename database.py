from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# import urllib.parse
 
# # Database connection parameters
# DB_USER = 'your_username'
# DB_PASSWORD = '@b#!$#6k'
# DB_HOST = 'localhost'
# DB_NAME = 'your_database_name'
 
# # Encode the password
# encoded_password = urllib.parse.quote(DB_PASSWORD, safe='')
 
# # Construct the connection string
# connection_string = f'postgresql://{DB_USER}:{encoded_password}@{DB_HOST}/{DB_NAME}'

SQLALCHEMY_DATABASE_URL = 'postgresql://postgres:postgres1234@localhost:5432/postgres'

engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()