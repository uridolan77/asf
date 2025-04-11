#!/usr/bin/env python3
'''
DSPy Learn Scraper using Selenium
'''
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import time
def scrape_with_selenium():
DSPy Learn Scraper using Selenium
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    # Create the driver
    driver = webdriver.Chrome(options=chrome_options)
    try:
        # Navigate to the page
        print("Opening https://dspy.ai/learn/")
        driver.get("https://dspy.ai/learn/")
        # Wait for the page to load
        time.sleep(5)
        # Get the page content
        content = driver.find_element(By.TAG_NAME, "body").text
        # Save the content
        with open("dspy_learn_selenium.txt", "w", encoding="utf-8") as f:
            f.write(content)
        print("Content saved to dspy_learn_selenium.txt")
    finally:
        # Close the browser
        driver.quit()
if __name__ == "__main__":
    scrape_with_selenium()