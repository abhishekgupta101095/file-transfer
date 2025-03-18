import requests
from bs4 import BeautifulSoup

# Define the target URL
url = "https://infopost.bwpipelines.com/Posting/default.aspx?Mode=Display&Id=11&tspid=1"

# Start a session
session = requests.Session()

# Step 1: Get the page content
response = session.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Step 2: Extract necessary ASP.NET hidden fields
viewstate = soup.find("input", {"name": "__VIEWSTATE"})["value"]
event_validation = soup.find("input", {"name": "__EVENTVALIDATION"})["value"]

# Step 3: Prepare the payload for the POST request
payload = {
    "__VIEWSTATE": viewstate,
    "__EVENTVALIDATION": event_validation,
    # "__EVENTTARGET": "dgITMatrix:9:lnkBtnDocumentTitle",  # Extracted from the `onclick`
    "__EVENTTARGET": 'dgITMatrix_8_lnkBtnDownload',
    "__EVENTARGUMENT": "",
}

# Step 4: Send the post request to trigger the download
download_response = session.post(url, data=payload)
print('dr',download_response.headers.get("Content-Type", ""))
print('dr',download_response.content)
# Step 5: Save the PDF (if response contains the file)
if "text/html" in download_response.headers.get("Content-Type", ""):
    with open("downloaded_file.xlsx", "wb") as f:
        f.write(download_response.content)
        print(download_response.content)
    print("PDF downloaded successfully!")
else:
    print("No PDF found in response. Check if additional authentication is needed.")
