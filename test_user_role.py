import pytest
from fastapi.testclient import TestClient
from starlette import status
from app.main import app

 
client = TestClient(app)

def test_exp_output():
    payload = {
        "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwidXNlcl9tYWlsIjoibi5rYXVzaGlrQGFjY2VudHVyZS5jb20iLCJpYXQiOjE1MTYyMzkwMjJ9.xb86zjB0wW3Lb4jg9_pc4L1JvtDqI60g7RQAbNzx38A",
        "key": "12345678",
        "algorithm": "string"
    }
    response2 = client.post('/User_Role_Fetch/', json=payload)

    print(response2.json())

    assert response2.status_code == 200
    assert response2.json() == [
  "80092ba6-0267-46b8-ac85-db77d6d0b5a7",
  "d61b98cc-191c-47b8-898d-5f0444a6842d"
]
 
def test_empty_jwt_token():
    payload = {
        "token": "",
        "key": "",
        "algorithm": "string"
    }
    response = client.post('/User_Role_Fetch/', json=payload)

    assert response.status_code == 400
    assert response.json() == {"detail": "The token can't be null."}



def test_empty_secret_key():
    payload = {
        "token": "xyz",
        "key": "",
        "algorithm": "string"
    }
    response = client.post('/User_Role_Fetch/', json=payload)

    print(response.json())

    assert response.status_code == 400
    assert response.json() == {"detail": "Please provide the secret key."}

def test_invalid_token():
    payload = {
        "token": "xyz",
        "key": "1",
        "algorithm": "string"
    }
    response = client.post('/User_Role_Fetch/', json=payload)

    print(response.json())

    assert response.status_code == 400
    assert response.json() == {"detail": "The token you provided is invalid."}

def test_data_not_exist():
    payload = {
        "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwidXNlcl9tYWlsIjoibkBhY2NlbnR1cmUuY29tIiwiaWF0IjoxNTE2MjM5MDIyfQ.l4wUYAqDXr4hRFlaytS78XDf-70AHf1rWe71E4QZlbA",
        "key": "12345678",
        "algorithm": "string"
    }
    response = client.post('/User_Role_Fetch/', json=payload)

    print(response.json())

    assert response.status_code == 404
    assert response.json() == {"detail": "The data you requested for doesnot exist."}


