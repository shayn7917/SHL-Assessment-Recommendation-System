import time, re
import pandas as pd
import requests as rq
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE = "https://www.shl.com/solutions/products/product-catalog/"

def list_cards(url: str):
    html = rq.get(url, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.select("a[href*='/product/'], a[href*='/solutions/']")
    out = []
    for a in anchors:
        href = urljoin(url, a.get("href") or "")
        title = a.get_text(" ", strip=True)
        if not title or "pre-packaged" in href.lower():
            continue
        out.append((title, href))
    seen, rows = set(), []
    for t, h in out:
        if h in seen:
            continue
        seen.add(h)
        rows.append((t, h))
    return rows

def fetch_detail(url: str):
    html = rq.get(url, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")
    ps = " ".join(p.get_text(" ", strip=True) for p in soup.select("p"))
    txt = re.sub(r"\s+", " ", ps).strip()[:6000]
    return {"description": txt}

def main():
    cards = list_cards(BASE)
    rows = []
    for title, href in cards:
        try:
            det = fetch_detail(href)
            rows.append({
                "assessment_name": title,
                "url": href,
                "description": det["description"]
            })
            time.sleep(0.3)
        except Exception as e:
            print("skip:", href, e)
    df = pd.DataFrame(rows).drop_duplicates(subset=["url"]).reset_index(drop=True)
    p = df["description"].str.contains(r"(personality|behaviou?r|motivation|style)", flags=re.I, regex=True)
    df["test_type"] = p.map({True:"P", False:"K"})
    df["tags"] = ""
    df.to_csv("data/processed/catalog.csv", index=False)
    print("saved:", len(df))

if __name__ == "__main__":
    main()
