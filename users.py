from typing import List
from fastapi import HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select
from starlette import status
from app.db import models
import app.schemas as schemas
from fastapi import APIRouter, FastAPI
# from fastapi import FastAPI
from app.db.database import get_db
# from db.models import User
import jwt
from jwt.algorithms import get_default_algorithms

get_default_algorithms()

router = APIRouter()    

@router.post('/getUserRole/')
def user_role_fetch(post_post:schemas.CreatePost, db:Session = Depends(get_db)):

    jwt_token = post_post.token
    secret_key = post_post.key

    if not jwt_token or jwt_token.strip()=='':
        print('** jwt',jwt_token)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The token can't be null.")
    if not secret_key or secret_key.strip()=='':
        print('$$sec',secret_key)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Please provide the secret key.")

    try:
        decoded_token = jwt.decode(jwt_token, secret_key, algorithms=['HS256'])
        user_mail = decoded_token['user_mail']

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"The signature of the token you provided is expired.")
    
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"The token you provided is invalid.")
    
    try:
        user_id = db.query(models.User).filter(models.User.email == user_mail).first().id
        user_type_id = db.query(models.User).filter(models.User.email == user_mail).first().user_type_id

    except AttributeError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"The data you requested for doesnot exist.")
    
    return user_id, user_type_id