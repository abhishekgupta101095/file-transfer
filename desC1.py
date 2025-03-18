import requests
from bs4 import BeautifulSoup

# Base URL
url = 'https://infopost.bwpipelines.com/Posting/default.aspx?Mode=Display&Id=11&tspid=1'

# Start a session
session = requests.Session()

# Fetch the first page
response = session.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Logic to navigate to the second page
# This may involve analyzing the form data or JavaScript that controls pagination
# For example, if there's a form with a specific action or hidden input fields

# Example: If the second page can be accessed via a specific URL pattern
second_page_url = url + '&Page=2'  # This is a hypothetical example
response = session.get(second_page_url)
soup = BeautifulSoup(response.content, 'html.parser')
print('soup',soup)
# Locate the table containing the postings
# table = soup.find('table', {'class': 'data-table'})  # Adjust the class or id as necessary
# print('table',table)
# # Extract rows 26 to 50
# rows = table.find_all('tr')[25:50]  # Note: list indices start at 0

# for row in rows:
#     # Extract and process data from each row
#     columns = row.find_all('td')
#     data = [col.get_text(strip=True) for col in columns]
#     print(data)
# Find the table
table = soup.find("table", {"id": "dgITMatrix"})

if not table:
    print("Table not found!")
    exit()

# Extract data
data = []
rows = table.find_all("tr", class_=lambda x: x and x.startswith("DataGrid-"))  # Get all relevant rows

for row in rows:
    cols = row.find_all("td")
    if len(cols) >= 3:
        description = cols[0].text.strip()
        date_time = cols[1].text.strip()
        download_link = cols[2].find("a")["href"] if cols[2].find("a") else "No Link"
        
        data.append({"Description": description, "Date/Time": date_time, "Download Link": download_link})

# Print results
for entry in data:
    print('e',entry)
