import pytest
from fastapi.testclient import TestClient
from starlette import status
from app.main import app

 
client = TestClient(app)
 
def test_empty_jwt_token():
    payload = {
        "jwt_token": "",
        "secret_key": "",
        "algorithm": "string"
    }
    response = client.post('/User_Role_Fetch/', json=payload)
    print(response.json())
    print(response.status_code)
    assert response.status_code == 400
    # assert response.json() == {"detail": "The token can't be null."}

# def test_invalid_jwt_token():
#     payload = {
#         "jwt_token": "xyz",
#         "secret_key": "abc",
#         "algorithm": "string"
#     }
#     response = client.post('/User_Role_Fetch/', json=payload)
#     # print(response.json())
#     # print(response.status_code)
#     assert response.status_code == 404
#     assert response.json() == {"detail": "The token can't be null."}
