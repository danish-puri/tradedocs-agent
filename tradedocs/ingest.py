"""Download the doc pages in sources.txt, split them into chunks, and build
the vector store and the endpoint list.

Run it with:  python -m tradedocs.ingest
Add --offline to reuse pages already downloaded into data/docs/.
"""

from __future__ import annotations

import argparse
import json
import sys

import requests

from .config import DOCS_DIR, ENDPOINTS_FILE, read_sources
from .docs import chunk_page, page_slug, parse_page
from .store import build_index


def fetch_pages(urls: list[str], offline: bool) -> dict[str, str]:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    pages = {}
    for url in urls:
        path = DOCS_DIR / f"{page_slug(url)}.md"
        if offline:
            if not path.exists():
                sys.exit(f"{path} is missing. Run once without --offline.")
        else:
            # Alpaca serves every docs page as plain markdown at <url>.md
            resp = requests.get(f"{url}.md", headers={"User-Agent": "tradedocs-agent/0.1"}, timeout=30)
            resp.raise_for_status()
            path.write_text(resp.text)
        pages[url] = path.read_text()
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="use pages already in data/docs/")
    args = parser.parse_args()

    urls = read_sources()
    pages = fetch_pages(urls, args.offline)

    all_chunks, all_endpoints = [], []
    for url, markdown in pages.items():
        page = parse_page(url, markdown)
        chunks, endpoints = chunk_page(page)
        all_chunks.extend(chunks)
        all_endpoints.extend(endpoints)
        print(f"{page.kind:9s} {page.slug:40s} {len(chunks):3d} chunks")

    ENDPOINTS_FILE.write_text(json.dumps(all_endpoints, indent=2))
    count = build_index(all_chunks)
    print(f"\nIndexed {count} chunks from {len(pages)} pages.")
    print(f"Saved {len(all_endpoints)} endpoints to {ENDPOINTS_FILE.relative_to(ENDPOINTS_FILE.parents[1])}.")


if __name__ == "__main__":
    main()
