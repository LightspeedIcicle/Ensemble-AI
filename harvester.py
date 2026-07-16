import os
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time

# CONFIGURATION
# Topics to hunt for. Add more as you see fit.
KEYWORDS = [
    "Casimir Effect",
    "Zero Point Energy",
    "Podkletnov",
    "Gravitomagnetism",
    "Alcubierre Drive",
    "Dynamic Nuclear Orientation"
    "Multi-Agent Reinforcement Learning",
    "Retrieval-Augmented Generation Architecture",
    "Agentic Workflow Patterns",
    "Vector Database Optimization",
    "Sovereign AI Infrastructure",
    "Local LLM Quantization Techniques",
    "Autonomous AI Research Agents",
    "Distributed System Design for AI"
    "4e AI"
    "INTP"
    "INTP 5w4 S/T"
    "Quiet BPD"
    "The Secrets of Birthdays"
            ]

MAX_RESULTS = 5  # Papers per keyword (Don't flood it yet)
SAVE_DIR = "knowledge"

def search_arxiv(query):
    """Searches ArXiv API for a specific keyword."""
    print(f"[*] Harvester: Scanning ArXiv for '{query}'...")
    base_url = 'http://export.arxiv.org/api/query?'
    search_query = f'all:{query}'
    
    # Safe URL encoding
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": MAX_RESULTS,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    url = base_url + urllib.parse.urlencode(params)
    
    try:
        response = urllib.request.urlopen(url)
        return response.read()
    except Exception as e:
        print(f"[!] Error contacting ArXiv: {e}")
        return None

def download_paper(url, title):
    """Downloads the PDF to the knowledge bunker."""
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    # Sanitize filename
    safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
    safe_title = safe_title.replace(" ", "_")
    filename = os.path.join(SAVE_DIR, f"{safe_title}.pdf")
    
    if os.path.exists(filename):
        print(f"    [-] Skipped (Already exists): {safe_title}")
        return

    print(f"    [+] Downloading: {safe_title}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print("        -> Secure.")
    except Exception as e:
        print(f"        -> FAILED: {e}")

def parse_and_harvest(xml_data):
    """Extracts PDF links from the XML response."""
    root = ET.fromstring(xml_data)
    # ArXiv uses an atom namespace
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    
    for entry in root.findall('atom:entry', ns):
        title = entry.find('atom:title', ns).text.strip()
        pdf_link = None
        
        for link in entry.findall('atom:link', ns):
            if link.attrib.get('title') == 'pdf':
                pdf_link = link.attrib['href']
        
        if pdf_link:
            download_paper(pdf_link, title)
            time.sleep(1) # Be polite to the server

def main():
    print("--- ENSCIO HARVESTER AGENT V1.0 ---")
    print(f"Targeting: {len(KEYWORDS)} Vectors")
    print("-----------------------------------")
    
    for keyword in KEYWORDS:
        data = search_arxiv(keyword)
        if data:
            parse_and_harvest(data)
        time.sleep(2)
        
    print("\n--- HARVEST COMPLETE ---")
    print(f"New Intel located in '{SAVE_DIR}/'.")
    print("Restart Aurora to ingest new data.")

if __name__ == "__main__":
    main()
