import os
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time

# CONFIGURATION
# Topics to hunt for. Every entry must be a term arXiv actually indexes.
#
# NOTE: keep the trailing commas. Without them Python silently concatenates
# adjacent string literals, and the harvester ends up searching for a keyword
# that does not exist instead of erroring.
KEYWORDS = [
    # Physics
    "Casimir Effect",
    "Zero Point Energy",
    "Podkletnov",
    "Gravitomagnetism",
    "Alcubierre Drive",
    "Dynamic Nuclear Orientation",
    # AI / systems
    "Multi-Agent Reinforcement Learning",
    "Retrieval-Augmented Generation Architecture",
    "Agentic Workflow Patterns",
    "Vector Database Optimization",
    "Sovereign AI Infrastructure",
    "Local LLM Quantization Techniques",
    "Autonomous AI Research Agents",
    "Distributed System Design for AI",
]

# Personal or exploratory topics belong in keywords.local.txt (gitignored,
# one keyword per line) rather than in this list, which is public.
LOCAL_KEYWORDS_FILE = "keywords.local.txt"

MAX_RESULTS = 5  # Papers per keyword (Don't flood it yet)
SAVE_DIR = "knowledge"

# arXiv's API terms ask for roughly one request every three seconds on a single
# connection. This is deliberately sequential and deliberately slow: parallelising
# it (e.g. with aiohttp) would get the client rate-limited or blocked, and this is
# a background batch job that is allowed to take its time.
REQUEST_DELAY_SECONDS = 3


def load_keywords():
    """Public keyword list, plus any private ones from keywords.local.txt."""
    keywords = list(KEYWORDS)
    if os.path.exists(LOCAL_KEYWORDS_FILE):
        with open(LOCAL_KEYWORDS_FILE, "r", encoding="utf-8") as f:
            extra = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        if extra:
            print(f"[*] Harvester: +{len(extra)} keyword(s) from {LOCAL_KEYWORDS_FILE}")
            keywords.extend(extra)
    return keywords

def search_arxiv(query):
    """Searches ArXiv API for a specific keyword."""
    print(f"[*] Harvester: Scanning ArXiv for '{query}'...")
    base_url = 'https://export.arxiv.org/api/query?'
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
        response = urllib.request.urlopen(url, timeout=30)
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
            time.sleep(REQUEST_DELAY_SECONDS)

def main():
    keywords = load_keywords()
    print("--- ENSCIO HARVESTER AGENT V1.0 ---")
    print(f"Targeting: {len(keywords)} Vectors")
    print("-----------------------------------")

    for keyword in keywords:
        data = search_arxiv(keyword)
        if data:
            parse_and_harvest(data)
        time.sleep(REQUEST_DELAY_SECONDS)

    print("\n--- HARVEST COMPLETE ---")
    print(f"New Intel located in '{SAVE_DIR}/'.")
    print("Restart Aurora to ingest new data.")

if __name__ == "__main__":
    main()
