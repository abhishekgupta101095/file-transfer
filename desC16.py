from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
from functools import lru_cache

# Constants
MAX_HITS = 30  # Limit website hits to 30
FILES_TO_DOWNLOAD = 25  # Total files to download (rows 26 to 50)
MAX_DOWNLOADS_PER_SESSION = 4  # Limit to 4 downloads per session
HITS_USED = 0  # Counter for website hits

# Caching function using LRU cache to limit hits
@lru_cache(maxsize=100)
def fetch_page(url):
    print(f"Fetching page {url} from the web...")
    response = requests.get(url)
    return response.text

# Function to start a browser session
def start_browser():
    chrome_options = webdriver.ChromeOptions()
    prefs = {
        "download.default_directory": os.getcwd(),  # Save downloads to current directory
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    chrome_options.add_experimental_option("prefs", prefs)
    return webdriver.Chrome(options=chrome_options)

# Function to download files while limiting hits and avoiding navigation error
def download_files():
    global HITS_USED
    # Open the website (1 hit)
    driver = start_browser()
    driver.get("https://infopost.bwpipelines.com/Posting/default.aspx?Mode=Display&Id=11&tspid=1")
    HITS_USED += 1

    # Navigate to the second page (1 hit)
    if HITS_USED < MAX_HITS:
        try:
            driver.execute_script("__doPostBack('dgITMatrix$_ctl4$_ctl1','')")
            time.sleep(3)  # Allow page to load
            HITS_USED += 1
        except Exception as e:
            print("Failed to navigate to page 2:", e)

    # Process batches of 4 downloads
    for batch_start in range(25, 50, MAX_DOWNLOADS_PER_SESSION):
        # Ensure that the hits don't exceed the limit
        if HITS_USED >= MAX_HITS:
            print("Hit limit reached. Stopping downloads.")
            break

        # Process the batch of up to 4 rows
        for row_num in range(batch_start, min(batch_start + MAX_DOWNLOADS_PER_SESSION, 50)):
            if HITS_USED >= MAX_HITS:
                print("Hit limit reached. Stopping downloads.")
                break

            try:
                element_id = f"dgITMatrix_{row_num}_lnkBtnDownload"
                download_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, element_id)))

                # Download the file (1 hit per click)
                download_button.click()
                print(f"Downloading file for row {row_num + 1}...")
                HITS_USED += 1

                # Allow time for the download to complete
                time.sleep(5)

            except Exception as e:
                print(f"Failed to download row {row_num + 1}: {e}")

        # Close the current browser session after 4 downloads
        driver.quit()

        # If there are more files to download, start a new browser session
        if HITS_USED < MAX_HITS:
            print(f"Starting new session after {batch_start + MAX_DOWNLOADS_PER_SESSION} downloads...")
            time.sleep(5)  # Pause before reopening browser
            driver = start_browser()
            driver.get("https://infopost.bwpipelines.com/Posting/default.aspx?Mode=Display&Id=11&tspid=1")
            HITS_USED += 1  # Revisit the page to continue from the second page
            driver.execute_script("__doPostBack('dgITMatrix$_ctl4$_ctl1','')")
            time.sleep(3)  # Allow page to load
            HITS_USED += 1  # Count the page navigation as a hit

    print(f"Total hits used: {HITS_USED}")
    driver.quit()  # Ensure browser is closed when done

# Run the function
download_files()
