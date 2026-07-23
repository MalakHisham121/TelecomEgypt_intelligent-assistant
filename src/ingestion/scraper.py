import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from typing import List
import logging

logger = logging.getLogger(__name__)

def scrape_te_page(url: str) -> List[Document]:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Ensure we read the response as UTF-8 for Arabic text
        response.encoding = 'utf-8'
        html_content = response.text
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # We can refine this to target specific divs or classes 
        # depending on the actual page structure (e.g., 'main' tag or specific IDs)
        # For now, we strip out scripts and styles to get clean text
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.decompose()
            
        text = soup.get_text(separator=' ', strip=True)
        
        # Create a single document for the page for now
        # We could also split this into multiple documents by heading if needed
        doc = Document(
            page_content=text,
            metadata={"source": url, "type": "web_page"}
        )
        
        logger.info(f"Successfully scraped {len(text)} characters from {url}")
        return [doc]
        
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return []

if __name__ == "__main__":
    # Test the scraper
    logging.basicConfig(level=logging.INFO)
    test_url = "https://te.eg/wps/portal/te/Personal"
    docs = scrape_te_page(test_url)
    if docs:
        print(f"Scraped sample content (first 200 chars): {docs[0].page_content[:200]}")
