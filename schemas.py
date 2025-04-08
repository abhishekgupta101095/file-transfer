# Defines Pydantic models for data validation and serialization.

from pydantic import BaseModel


class PostBase(BaseModel):
    token: str = ''
    key: str = ''
    algorithm:str = ''

    class Config:
        orm_mode = True


class CreatePost(PostBase):
    class Config:
        orm_mode = True