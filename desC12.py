import requests
from bs4 import BeautifulSoup

# URL of the page containing the links
URL = "https://infopost.bwpipelines.com/Posting/default.aspx?Mode=Display&Id=11&tspid=1"

# Start a session to maintain cookies and session data
session = requests.Session()

# Get the page content
response = session.get(URL)
soup = BeautifulSoup(response.text, "html.parser")
print('soup',soup)

# Extract hidden form fields required for the post request
viewstate = soup.find("input", {"name": "__VIEWSTATE"})["value"]
print('svps',soup.find("input", {"name": "__VIEWSTATE"}))
print('viewstate',viewstate)
eventvalidation = soup.find("input", {"name": "__EVENTVALIDATION"})["value"]

# Find all links triggering JavaScript downloads
pdf_links = soup.find_all("<a", onclick=True)
print('sfa')
print('pdfl',pdf_links)
# Iterate over found links and simulate the download
for link in pdf_links:
    onclick_text = link["onclick"]
    print('onec',onclick_text)
    if "WebForm_DoPostBackWithOptions" in onclick_text:
        
        # Extract the postback event target from the JavaScript function
        start = onclick_text.find("('") + 2
        end = onclick_text.find("',")
        event_target = onclick_text[start:end]
        
        # Prepare the POST request payload
        payload = {
            "__EVENTTARGET": event_target,
            "__EVENTARGUMENT": "",
            "__VIEWSTATE": viewstate,
            "__EVENTVALIDATION": eventvalidation,
        }
        
        # Send the POST request to trigger the download
        pdf_response = session.post(URL, data=payload)
        print('pdfres',pdf_response)
        # Save the PDF file
        pdf_filename = link.text.strip().replace(" ", "_") + ".pdf"
        with open(pdf_filename, "wb") as pdf_file:
            pdf_file.write(pdf_response.content)
        print(f"Downloaded: {pdf_filename}")
