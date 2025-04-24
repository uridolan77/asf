#!/usr/bin/env python3
DSPy Learn Fallback Scraper - Tries multiple methods to get content
import urllib3
import certifi
import ssl
import time
import os
from bs4 import BeautifulSoup
import warnings
# Suppress warnings
warnings.filterwarnings('ignore')
def scrape_dspy_learn():
    """
    scrape_dspy_learn function.
    
    This function provides functionality for..."""
    print("DSPy Learn Fallback Scraper")
    # Base URL
    base_url = "https://dspy.ai/learn/"
    # Try method 1: Custom SSL Context with urllib3
    print("\nMethod 1: Using urllib3 with custom SSL context...")
    content = try_urllib3_method(base_url)
    if content:
        save_content(content, "dspy_learn_urllib3.txt")
        return
    # Try method 2: HTTP instead of HTTPS
    print("\nMethod 2: Trying HTTP instead of HTTPS...")
    http_url = base_url.replace("https://", "http://")
    content = try_urllib3_method(http_url, use_ssl=False)
    if content:
        save_content(content, "dspy_learn_http.txt")
        return
    # Try method 3: Hardcoded content from documentation
    print("\nMethod 3: Using hardcoded content from documentation...")
    content = get_hardcoded_content()
    if content:
        save_content(content, "dspy_learn_hardcoded.txt")
        return
    print("\nAll methods failed. Please consider using a browser automation tool like Selenium.")
    print("Instructions for using Selenium:")
    print("1. Install Selenium: pip install selenium")
    print("2. Download webdriver for your browser")
    print("3. Use the Selenium sample code provided in this directory")
def try_urllib3_method(url, use_ssl=True):
    Try to fetch content using urllib3 with custom SSL settings.
    
    Args:
        url: Description of url
        use_ssl: Description of use_ssl
    
def save_content(content, filename):
    Save the content to a file.
    
    Args:
        content: Description of content
        filename: Description of filename
    
    with open("dspy_learn_selenium.py", "w", encoding="utf-8") as f:
        f.write(selenium_code)
    print("Created Selenium example script: dspy_learn_selenium.py")
if __name__ == "__main__":
    try:
        scrape_dspy_learn()
    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")