import requests
from bs4 import BeautifulSoup


# Base URL of the dataset
base_url = 'https://gats.pjm-eis.com/gats2/PublicReports/RenewableGeneratorsRegisteredinGATS'

session = requests.Session()


# Parameters for pagination
params = {
    'pageSize': 200,  # Set page size to 200
    'page': 2         # Access the second page
}

# Send GET request to fetch data
response = requests.get(base_url, params=params)
print("response",response)
# response = session.get(base_url)
# soup = BeautifulSoup(response.content, 'html.parser')
# print('soup',soup)
# Check if the request was successful
if response.status_code == 200:
    data = response.json()  # Assuming the response is in JSON format
    # Process the data as needed
else:
    print(f'Error: {response.status_code}')
