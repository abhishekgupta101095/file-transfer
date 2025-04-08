import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Test case for missing products
def test_create_organization_missing_products():
    #Arrange
    payload = {
        "name": "Test Organization",
        "primary_contact": "test@example.com",
        "products_name": ["Product A", "Product C"]
    }
    #Act
    response = client.post("/organizations/", json=payload)
    #Assert
    assert response.status_code == 400
    assert "The following product(s) are not found in the Products list: Product A, Product C" in response.json()["detail"]


# Test case for invalid email
def test_create_organization_invalid_email():
    payload = {
        "name": "Test Organization",
        "primary_contact": "invalid_email",
        "products_name": ["Product A", "Product B"]
    }
    response = client.post("/organizations/", json=payload)
    assert response.status_code == 400
    assert "Invalid email format for primary contact" in response.json()["detail"]

def test_create_organization_success_with_product_APIs():
    payload = {
        "name": "Test Organization",
        "primary_contact": "test@example.com",
        "products_name": ["CRISP"]
    }
    response = client.post("/organizations/", json=payload)
    assert response.status_code == 200
    assert response.json() == {"message": "Created new Organization", "organization Name": "Test Organization"}


# Test case for valid email
def test_create_organization_valid_email():
    #Arrange
    payload = {
        "name": "Test Organization",
        "primary_contact": "test@example.com",
        "products_name": ["CRISP"]
    }
    #Act
    response = client.post("/organizations/", json=payload)
    #Assert
    assert response.status_code == 200
    assert response.json() == {"message": "Created new Organization", "organization Name": "Test Organization"}


# Test case for empty organization name
def test_create_organization_empty_name():
    payload = {
        "name": "",
        "primary_contact": "test@example.com",
        "products_name": ["CRISP"]
    }
    response = client.post("/organizations/", json=payload)
    assert response.status_code == 400
    assert "Organization name cannot be empty" in response.json()["detail"]


# Test case for duplicate organization
def test_create_organization_duplicate_organization():
    payload = {
        "name": "Test Organization",
        "primary_contact": "test@example.com",
        "products_name": ["CRISP"]
    }
    response = client.post("/organizations/", json=payload)
    assert response.status_code == 400
    assert "Organization with this name already exists" in response.json()["detail"]