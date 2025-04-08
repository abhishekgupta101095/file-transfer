from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, TIMESTAMP, Boolean, text, UUID

Base = declarative_base() # factory function that creates a new base class specifically designed for declarative mapping with SQLAlchemy.

class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer,primary_key=True,nullable=False)
    title = Column(String,nullable=False)
    content = Column(String,nullable=False)
    published = Column(Boolean, server_default='TRUE')
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))
    phone_number = Column(String,nullable=False)

class User(Base):
    __tablename__ = "users"

    id = Column(UUID,primary_key=True,nullable=False)
    email = Column(String,nullable=False)
    org_id = Column(UUID,nullable=True)
    user_type_id = Column(UUID,nullable=True)