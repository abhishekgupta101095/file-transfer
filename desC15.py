import requests

# URL of the ASP.NET page
url = "https://infopost.bwpipelines.com/Posting/default.aspx?Mode=Display&Id=11&tspid=1"  # Replace with actual URL

# Form data for the POST request
payload = {
    "__EVENTTARGET": "dgITMatrix:0:lnkBtnDownload",  # Target control
    "__EVENTARGUMENT": "",
    "__VIEWSTATE_PAGE_INDEX": 11,
    "__VIEWSTATE": "",  # Extract this from page source
    "__VIEWSTATEGENERATOR": "/wEWOAL+raDpAgLYhvrYBwLXkqeHCgLyiNnrCAKTteWvDQLZ9LjiBAKpqZfpCwLO4OcUAqWa//YHArmbm6wGAubR390KAsKR17UMAqjy4JkLAq3nk5EPAtyvh7gPAqLTguwBArzomm0C05qJzwoC+equwQ8C/qLhnQ0Ck7CViQYC2Y/C2Q8Cma3qyAsCzqOnmAECo/Oz0AwCsOTCnA0ChvH7/wsCj+zInA0C4ZCbrQwC7rPGnA0C7Ibn/Q0C9bu9nA0Cz+Lrjw0C3P/JnA0C0tyzxw4Cu4fQnA0CjYCChQ8Cms3NnA0CyI/jngwC0drEnA0Cm8rMhA0CyLfYnA0C/oXbsQ0Cp/3VnA0Cm9rW5wEC4aTz/w0C/teelwECwKz5/w0C2fe9xAECn/T2/w0C5O2JlQMCpvzt/w0Cx8mOpwICjcD6/w0C8oid4QwC6uKW/A3FJOTmr/enpjYemby/KU7yrZNFdQ==",  # Extract from page source
}

# Start a session to maintain cookies
session = requests.Session()

# Get initial page to retrieve VIEWSTATE and EVENTVALIDATION values
response = session.get(url)
if response.status_code == 200:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extract necessary hidden fields
    # payload["__VIEWSTATE"] = soup.find("input", {"name": "__VIEWSTATE"})["value"]
    # payload["__VIEWSTATEGENERATOR"] = soup.find("input", {"name": "__VIEWSTATEGENERATOR"})["value"]
    payload = {
    "__EVENTTARGET": "dgITMatrix:0:lnkBtnDownload",  # Target control
    "__EVENTARGUMENT": "",
    "__VIEWSTATE_PAGE_INDEX": 11,
    "__VIEWSTATE": "",  # Extract this from page source
    "__VIEWSTATEGENERATOR": "/wEWOAL+raDpAgLYhvrYBwLXkqeHCgLyiNnrCAKTteWvDQLZ9LjiBAKpqZfpCwLO4OcUAqWa//YHArmbm6wGAubR390KAsKR17UMAqjy4JkLAq3nk5EPAtyvh7gPAqLTguwBArzomm0C05qJzwoC+equwQ8C/qLhnQ0Ck7CViQYC2Y/C2Q8Cma3qyAsCzqOnmAECo/Oz0AwCsOTCnA0ChvH7/wsCj+zInA0C4ZCbrQwC7rPGnA0C7Ibn/Q0C9bu9nA0Cz+Lrjw0C3P/JnA0C0tyzxw4Cu4fQnA0CjYCChQ8Cms3NnA0CyI/jngwC0drEnA0Cm8rMhA0CyLfYnA0C/oXbsQ0Cp/3VnA0Cm9rW5wEC4aTz/w0C/teelwECwKz5/w0C2fe9xAECn/T2/w0C5O2JlQMCpvzt/w0Cx8mOpwICjcD6/w0C8oid4QwC6uKW/A3FJOTmr/enpjYemby/KU7yrZNFdQ==",  # Extract from page source
}

    # Send POST request to trigger the file download
    download_response = session.post(url, data=payload)
    print('dr',download_response.content)
    # Save the file if response is valid
    if "Content-Disposition" in download_response.headers:
        with open("downloaded_file.ext", "wb") as file:
            file.write(download_response.content)
        print("Download successful!")
    else:
        print("Failed to download file. Check response content.")