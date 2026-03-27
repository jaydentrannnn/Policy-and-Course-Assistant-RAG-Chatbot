"""
Web Crawler - crawls a domain and saves every page to disk.

Usage:
    python crawler.py https:example.com
    python crawler.py https://example.com --output ./pages --delay 1.0 --max 200
"""


import argparse
import os
from os import path
import re
import time
from datetime import datetime, timezone
import urllib.parse
from collections import deque
from pathlib import Path

import requests
from bs4 import BeautifulSoup


#────────────────────────────────────────────────────────────────────────────────────────────
# Helper Functions
#────────────────────────────────────────────────────────────────────────────────────────────

def normalize_url(url: str) -> str:
    """Removing fragments and query parameters."""
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.strip('/')
    return parsed._replace(fragment='', query='', path=path).geturl()


def url_to_filepath(base_dir: Path, url: str) -> Path:
    """Convert a URL to a local .json file path under base_dir."""
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.strip('/')

    # Query strings become part of the filename
    if parsed.query:
        safe_query = re.sub(r"[^a-zA-Z0-9_\-]", "_", parsed.query)
        path = f"{path}__{safe_query}"

    if parsed.fragment:
        safe_fragment = re.sub(r"[^a-zA-Z0-9_\-]", "_", parsed.fragment)
        path = f"{path}__{safe_fragment}"
    
    if not path or path.endswith('/'):
        path += 'index' # treat directory paths as index.json
    elif "." not in Path(path).name:
        path += '/index' # treat paths without file extension as directories
    else:
        path = str(Path(path).with_suffix('')) # remove file extension if present

    return base_dir / (path + ".json")


def is_allowed_url(url: str, origin: str) -> bool:
    parsed_url    = urllib.parse.urlparse(url)
    parsed_origin = urllib.parse.urlparse(origin)

    # Must be the same domain
    if parsed_url.netloc != parsed_origin.netloc:
        return False

    # Must be under the same path prefix
    # e.g. origin path "/allcourses" — only allow URLs starting with "/allcourses"
    origin_prefix = parsed_origin.path.rstrip('/')
    return parsed_url.path.startswith(origin_prefix)


#────────────────────────────────────────────────────────────────────────────────────────────
# Parser
#────────────────────────────────────────────────────────────────────────────────────────────

METADATA_LABELS = [
    "Prerequisite",
    "Corequisite",
    "Restriction",
    "Restrictions",
    "Grading Option",
    "Repeatability",
    "Concurrent with",
    "Overlaps with",
    "Same as",
]

def parse_courses(soup: BeautifulSoup, page_url: str):
    """Detect and parse UCI-style course blocks from a page."""
    blocks = soup.find_all('div', class_='courseblock')
    if not blocks:
        return None
    
    courses = []
    crawled_at = datetime.now().isoformat()
    
    for block in blocks:
        course = {"url": page_url, "crawled_at": crawled_at}

        # 1. Extract Code, Title, and Units
        code_tag = block.find('span', class_=re.compile(r'detail-code'))
        title_tag = block.find('span', class_=re.compile(r'detail-title'))
        hours_tag = block.find('span', class_=re.compile(r'detail-hours_html'))

        course["code"]  = code_tag.get_text(strip=True).rstrip('.') if code_tag else ""
        course["title"] = title_tag.get_text(strip=True).rstrip('.') if title_tag else ""
        
        course["units"] = ""
        if hours_tag:
            # Handles single numbers like "4" and ranges like "1-4"
            m = re.search(r"(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)\s+Units?", hours_tag.get_text(strip=True), re.IGNORECASE)
            if m:
                course["units"] = m.group(1) 
        
        # 2. Extract Description 
        desc_div = block.find('div', class_='courseblockextra')
        course["description"] = desc_div.get_text(" ", strip=True) if desc_div else ""

        # 3. Extract Metadata (Prereqs, Restrictions, etc.)
        metadata = {}
        for label_tag in block.find_all('span', class_='label'):
            label = label_tag.get_text(strip=True).rstrip(':').strip()
            parent = label_tag.parent
            if parent:
                label_tag.extract() # Remove the label from the HTML to isolate the value
                value = parent.get_text(" ", strip=True)
                metadata[label] = value

        course["prerequisite"]   = metadata.get("Prerequisite", "")
        course["corequisite"]    = metadata.get("Corequisite", "")
        course["restrictions"]   = metadata.get("Restrictions", metadata.get("Restriction", ""))
        course["grading_option"] = metadata.get("Grading Option", "")
        course["repeatability"]  = metadata.get("Repeatability", "")

        known = {"Prerequisite", "Corequisite", "Restrictions", "Restriction", "Grading Option", "Repeatability"}
        course["extra"] = {k: v for k, v in metadata.items() if k not in known}
                
        courses.append(course)

    return courses



def extract_page_data(html: str, url: str) -> dict:
    """Extract relevant data from the HTML content of a page."""
    soup = BeautifulSoup(html, 'html.parser')

    courses = parse_courses(soup, url)
    if courses is not None:
        return {"type": "course_page", "url": url, "courses": courses,
                "crawled_at": datetime.now(timezone.utc).isoformat()}
    

    title = soup.title.string if soup.title else ''
    # Meta description
    description = ""
    meta = soup.find('meta', attrs={'name': 'description'})
    if meta and meta.get('content'):
        description = meta['content'].strip()

    headings = []
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        text = tag.get_text(strip=True)
        if text:
            headings.append(text)

    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()  # Remove script and style tags
    body_text = ' '.join(soup.get_text(separator=' ', strip=True).split())

    links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        abs_url = urllib.parse.urljoin(url, href)
        if urllib.parse.urlparse(abs_url).scheme in ("http", "https"):
            links.append(abs_url)
 
    return {
        "type": "page",
        "url": url,
        "title": title,
        "description": description,
        "headings": headings,
        "text": body_text,
        "links": links,
        "crawled_at": datetime.now(timezone.utc).isoformat(),
    }
 
 
def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        abs_url = urllib.parse.urljoin(base_url, href)
        if urllib.parse.urlparse(abs_url).scheme in ("http", "https"):
            links.append(abs_url)
    return links


def crawl(start_url: str, base_dir: Path = Path("./crawled_pages"), delay: float = 0.5, max_pages: int = 1000, timeout: int = 10, max_retries: int = 4) -> None:
    """Crawl a website starting from start_url, saving pages to base_dir."""
    start_url = normalize_url(start_url)
    base_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; PyCrawler/1.0)"})

    visited: set[str] = set()
    queue: deque[str] = deque([start_url])
    retry_counts: dict[str, int] = {}

    print(f"Starting crawl at {start_url}")
    print(f"Saving pages to {base_dir.resolve()}")
    print(f"Delay between requests: {delay} seconds")
    print(f"Maximum pages to crawl: {max_pages}")

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        url = normalize_url(url)

        if url in visited:
            continue
        if not is_allowed_url(url, start_url):
            continue

        visited.add(url) # Mark as visited before crawling to avoid duplicates in the queue

        try:
            response = session.get(url, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            html = response.text

            data = extract_page_data(html, url)
            filepath = url_to_filepath(base_dir, url)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                import json
                json.dump(data, f, ensure_ascii=False, indent=2)

            visited.add(url)
            print(f"Crawled ({len(visited)}/{max_pages}): {url}")

            links = extract_links(html, url)
            for link in links:
                if link not in visited and is_allowed_url(link, start_url):
                    queue.append(link)

        except requests.RequestException as e:
            visited.remove(url)
            attempts = retry_counts.get(url, 0)
            if attempts < max_retries:
                retry_counts[url] = attempts + 1
                queue.append(url)
                print(f"Failed {url}: {e}. Pushing to back of queue (Retry {attempts + 1}/{max_retries}).")
            else:
                visited.add(url) # Permanently mark as visited so we stop trying
                print(f"Gave up on {url} after {max_retries} retries.")

        time.sleep(delay)
    
    print(f"Crawl finished. Total pages crawled: {len(visited)}")


def main():
    parser = argparse.ArgumentParser( 
        description="Crawl a domain and save all pages as JSON."
    )
    parser.add_argument("url", help="Starting URL (e.g. https://catalogue.uci.edu/allcourses)")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Directory to save JSON files (default: ./<domain>)"
    )
    parser.add_argument(
        "--delay", "-d", type=float, default=0.5,
        help="Seconds to wait between requests (default: 0.5)"
    )
    parser.add_argument(
        "--max", "-m", type=int, default=500,
        help="Maximum number of pages to crawl (default: 500)"
    )
    parser.add_argument(
        "--timeout", "-t", type=int, default=10,
        help="Request timeout in seconds (default: 10)"
    )
    args = parser.parse_args()
 
    output_dir = args.output or urllib.parse.urlparse(args.url).netloc or "crawled_pages"
 
    crawl(
        start_url=args.url,
        base_dir=Path(output_dir),
        delay=args.delay,
        max_pages=args.max,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
