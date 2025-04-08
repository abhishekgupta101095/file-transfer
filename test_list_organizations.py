import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.db.database import get_db
from app.db import models
from app.db.models import Base

SQLALCHEMY_DATABASE_URL = 'postgresql://<username>:<password>@<hostname>:<port>/<test_dbname>'
test_engine = create_engine(SQLALCHEMY_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

Base.metadata.create_all(bind=test_engine)  

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()  
            
app.dependency_overrides[get_db] = override_get_db


@pytest.fixture()
def test_db():
    try:
        db = TestingSessionLocal()
        yield  db
    finally:
        db.close()

@pytest.fixture()
def test_products(test_db): 
    product = models.Products(
        id = '8bcc2ba4-297e-4212-94d5-9a846d174492',
        name = "Test Product-" + str(uuid.uuid4()),
        description = "",
        create_user_api = "",
        edit_user_api = "",
        delete_user_api = "",
        is_teams_required = False,
        create_team_api = "",
        edit_team_api = "",
        delete_team_api = "",
        list_team_api = "",
        is_custom_role_required = False,
        list_custom_role_api = "",
        create_org_api = ""
    )
    test_db.add(product)
    test_db.commit()
    test_db.refresh(product)
    return product

@pytest.fixture()
def test_organization(test_db):
    organization = models.Organization(
        id = uuid.uuid4(), # 'a74a8f97-1c19-4efa-89f5-e2bca94c1dee',
        name = "Test Org-" + str(uuid.uuid4()),
        products_enabled = ['8bcc2ba4-297e-4212-94d5-9a846d174492'],
        primary_contact = "test_contact@example.com"
    )
    test_db.add(organization)
    test_db.commit()
    test_db.refresh(organization)
    return organization

@pytest.fixture()
def test_admin_user(test_db, test_organization):
    user = models.User(
        id = uuid.uuid4(),# '80092ba6-0267-46b8-ac85-db77d6d0b5a7',
        email = "test_contact@example.com",
        org_id = test_organization.id,
        user_type_id = 'd61b98cc-191c-47b8-898d-5f0444a6842d'  # Set a UUID for either the 'Admin' or 'Accenture Admin' type
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user


client = TestClient(app)


# -------------------- Test case 1 ------------------ #

def test_get_organization_success(test_admin_user, test_organization, test_products, test_db):
    
    response = client.get(
        "/organizations/",
        params={"user_id": str(test_admin_user.id), "user_type_id": str(test_admin_user.user_type_id)}
    )
    
    data = response.json()
    assert response.status_code == 200
    
    user = test_db.query(models.User).filter(models.User.id == test_admin_user.id).first() 
    db_organization = user.organization
    product_ids = db_organization.products_enabled
    expected_product_names = [
        test_db.query(models.Products).filter(models.Products.id == product_id).first().name
        for product_id in product_ids
    ]

    assert data["organization"] == db_organization.name
    assert data["product"] == expected_product_names
    assert data["users_count"] == test_db.query(models.User).filter(models.User.org_id == user.organization.id).count()
    assert data["primary_contact"] == db_organization.primary_contact
    

# -------------------- Test case 2 ------------------ #
   
def test_get_organization_unauthorized(test_db):
    # Fetch user and user_type_id belonging to different records
    user_1 = test_db.query(models.User).first()
    
    # Then get another user ensuring a different user_type_id
    user_2 = test_db.query(models.User).filter(models.User.user_type_id != user_1.user_type_id).first()

    response = client.get(
        "/organizations/",
        params={"user_id": str(user_1.id), "user_type_id": str(user_2.user_type_id)} 
    )

    assert response.status_code == 404


# -------------------- Test case 3 ------------------ #

def test_get_organization_forbidden(test_db):
    # Fetch a user who is NOT an 'Accenture Admin' 
    user_1 = test_db.query(models.User).join(models.UserType).filter(models.UserType.name != "Accenture Admin").first()

    response = client.get(
        "/organizations/",
        params={"user_id": str(user_1.id), "user_type_id": str(user_1.user_type_id)} 
    )
    
    assert response.status_code == 403


# -------------------- Test case 4 ------------------ #
    
def test_get_organization_user_not_found(test_db):
    invalid_user_id = uuid.uuid4()
    valid_user_type_id = test_db.query(models.UserType).first().id  # Fetch an existing user_type_id

    response = client.get(
        "/organizations/",
        params={"user_id": str(invalid_user_id), "user_type_id": str(valid_user_type_id)} 
    )

    assert response.status_code == 404