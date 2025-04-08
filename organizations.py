# app/routers/organizations.py

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from db.models import Organization, Products
from schemas import OrganizationCreate
from uuid import UUID
from fastapi import HTTPException
import re
import requests
from sqlalchemy import or_
from starlette import status
from db import models
import schemas as schemas
from db.database import get_db

router = APIRouter()

headers = {

    "Authorization": "bearer token",
   
    "Content-Type": "application/json"
}

# API to list organizations
@router.get("/organizations/", response_model=schemas.OrganizationData)
def list(user_id: UUID, user_type_id: UUID, db: Session = Depends(get_db)):
    # Authorization logic
    user = (
    db.query(models.User)
    .filter(models.User.id == user_id, models.User.user_type_id == user_type_id)
    .first()
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    if user.user_type.name != "Accenture Admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access",
        )    

    product_ids = user.organization.products_enabled
    products = db.query(models.Products).filter(models.Products.id.in_(product_ids)).all()
    product_names = [product.name for product in products]

    return {
        "organization": user.organization.name,
        "product" : product_names,
        "users_count": db.query(models.User)
                         .filter(models.User.org_id == user.organization.id)
                         .count(),
        "primary_contact": user.organization.primary_contact,
    }


# API to create new organization
@router.post("/organizations/")
def create_organization(organization: OrganizationCreate, db: Session = Depends(get_db)):
    # Check if the organization name is empty
    if not organization.name:
        raise HTTPException(status_code=400, detail="Organization name cannot be empty")
    
    # Check if the organization name already exists
    existing_organization = db.query(Organization).filter(Organization.name == organization.name).first()
    if existing_organization:
        raise HTTPException(status_code=400, detail="Organization with this name already exists")

    EMAIL_REGEX = re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$')
    if not re.match(EMAIL_REGEX, organization.primary_contact):
        raise HTTPException(status_code=400, detail="Invalid email format for primary contact")
    
    products = db.query(Products).filter(or_(*[Products.name.ilike(f"%{product_name}%") for product_name in organization.products_name])).all()
    
    # Storing in organizations table with product id instead of product name
    product_ids = []
    missing_products = [] 
    product_urls = {} 
    for product in products:
        product_ids.append(product.id)
        # Storing product urls for calling product APIs
        product_urls[product.name] = product.create_org_api
    
    
    # Check if any products are missing
    missing_products = set(organization.products_name) - set([product.name for product in products])
    if missing_products:
        raise HTTPException(status_code=400, detail=f"The following product(s) are not found in the Products list: {', '.join(missing_products)}")
    


    # Calling all product APIs that new organization is created
    hostname = "https://2q2amqd2r9.execute-api.us-east-1.amazonaws.com/dev/"
    for product_name, url in product_urls.items():
        # As we have only CRISP API
        if url is not None:
            url = hostname + url
            
            # Payload for product API
            payload = {
                "name": organization.name,
                "description": "New organization is created",
                "members": [{"email": organization.primary_contact, "leader": 1}]
            }      
            response = requests.post(url, json=payload, headers=headers)
            
            # If any product API fails, not storing the new organization in the database
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Error posting to product API {product_name}")
        else:
            raise HTTPException(status_code=400, detail=f"Product API URL not found for product {product_name}")
    
    # Storing new organization only if all product APIs are successful
    db_organization = Organization(
        name=organization.name,
        products_enabled=product_ids,
        primary_contact=organization.primary_contact
    )
    db.add(db_organization)
    db.commit()
    db.refresh(db_organization)

    return {"message": "Created new Organization", "organization Name": organization.name}