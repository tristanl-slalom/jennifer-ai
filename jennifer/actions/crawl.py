from collections import deque
from pathlib import Path
from shutil import rmtree
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from jennifer.utilities.hyperlinks import get_domain_hyperlinks


def crawl_action(url: str, rebuild: bool, must_include: str):

    local_domain = urlparse(url).netloc

    queue = deque([url])

    seen = {url}

    output_path = Path("output")
    text_domain_dir = output_path / "text" / local_domain

    if text_domain_dir.exists():
        if not rebuild:
            return
        else:
            rmtree(text_domain_dir)

    text_domain_dir.mkdir(exist_ok=True, parents=True)

    while queue:
        url = queue.pop()
        print(f"{len(queue)}: {url}")

        santized_url_sans_protocol = url[8:].replace("/", "__").replace(":", "--")[:64]
        with open(
            text_domain_dir / f"{santized_url_sans_protocol}.txt", "w", encoding="utf-8"
        ) as f:
            try:
                raw_page = requests.get(url, headers={"User-Agent": "XY"})
                soup = BeautifulSoup(raw_page.text, "html.parser")

                text = soup.get_text()

                if "You need to enable JavaScript to run this app." in text:
                    print(f"Unable to parse page {url} due to JavaScript requirements")

                f.write(text)
            except requests.exceptions.ConnectionError:
                continue

        for link in get_domain_hyperlinks(local_domain, url):
            if link not in seen and (not must_include or must_include in link):
                queue.append(link)
                seen.add(link)
