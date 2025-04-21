

import requests, re, json, time, os
from bs4 import BeautifulSoup

BASE_URL    = "https://www.shl.com"
CATALOG_URL = BASE_URL + "/solutions/products/product-catalog/"
DATA_DIR    = "data"
OUTPUT_JSON = os.path.join(DATA_DIR, "assessments4.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def fetch_soup(url, params=None):
    resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")

def collect_all_product_urls():
    """
    Iterate server‑side pages for type=2 (pre‑packaged) and type=1 (individual),
    grabbing every unique /view/.../ link.
    """
    pattern = re.compile(r"^/solutions/products/product-catalog/view/.+?/$")
    seen = set()
    for type_val in (2, 1):
        page = 0
        while True:
            params = {"type": type_val, "start": page * 12} if page > 0 else {"type": type_val}
            print(f"→ Fetching catalog page: type={type_val} start={params.get('start',0)}")
            soup = fetch_soup(CATALOG_URL, params=params)
            anchors = soup.find_all("a", href=pattern)
            new = 0
            for a in anchors:
                href = a["href"]
                full = href if href.startswith("http") else BASE_URL + href
                if full not in seen:
                    seen.add(full)
                    new += 1
            print(f"   found {len(anchors)} links, {new} new")
            if new == 0:
                break
            page += 1
            time.sleep(1)
    print(f"ℹ️ Total unique product URLs found: {len(seen)}")
    return list(seen)

def parse_detail_page(url):
    """
    Scrape one assessment detail page for:
      - name
      - description
      - duration_minutes
      - remote_testing (Yes/No)
      - adaptive_irt   (Yes/No)
      - test_types     (list of codes)
    """
    soup = fetch_soup(url)

    # Name
    name_tag = soup.find("h1")
    name = name_tag.get_text(strip=True) if name_tag else url.split("/")[-2].replace("-", " ").title()

    # Description: the <p> immediately under the "Description" heading
    desc = ""
    hdr = soup.find(lambda tag: tag.name in ["h2","h3","h4"] and "Description" in tag.get_text())
    if hdr:
        p = hdr.find_next_sibling("p")
        if p:
            desc = p.get_text(strip=True)

    # Duration: look for "minutes" in the text
    duration = ""
    txt = soup.find(text=re.compile(r"(\d+)\s*minutes", re.I))
    if txt:
        m = re.search(r"(\d+)\s*minutes", txt)
        duration = m.group(1) if m else ""

    # Remote Testing & Adaptive/IRT: presence of those phrases anywhere
    remote   = "Yes" if soup.find(text=re.compile(r"Remote Testing", re.I)) else "No"
    adaptive = "Yes" if soup.find(text=re.compile(r"Adaptive/?IRT", re.I)) else "No"

    # Test Type codes: everything after "Test Type:" that's a single uppercase letter
    # test_types = [
    #     span.get_text(strip=True)
    #     for span in soup.find_all("span", class_="product-catalogue__key")
    # ]

    return {
        "name":             name,
        "url":              url,
        "description":      desc,
        "duration_minutes": duration,
        "remote_testing":   remote,
        "adaptive_irt":     adaptive,
    }

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1) Gather every product URL
    urls = collect_all_product_urls()

    # 2) Scrape detail pages
    results = []
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] Scraping {url}")
        try:
            results.append(parse_detail_page(url))
        except Exception as e:
            print("  ❌ failed:", e)
        time.sleep(0.5)

    # 3) Write out
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! Scraped {len(results)} assessments → {OUTPUT_JSON}")

if __name__ == "__main__":
    main()


