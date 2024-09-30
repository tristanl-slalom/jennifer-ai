from collections import deque
from shutil import rmtree
from urllib.parse import urlparse
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from jennifer.get_domain_hyperlinks import get_domain_hyperlinks


def crawl_action(url: str, rebuild: bool, must_include: str):

    local_domain = urlparse(url).netloc

    queue = deque([url])

    seen = {url}

    text_dir = Path("text")
    text_domain_dir = text_dir / local_domain

    if text_domain_dir.exists():
        if not rebuild:
            return
        else:
            rmtree(text_domain_dir)

    text_domain_dir.mkdir(exist_ok=True, parents=True)

    while queue:
        url = queue.pop()
        print(url)

        santized_url_sans_protocol = url[8:].replace("/", "__")
        with open(text_domain_dir / f"{santized_url_sans_protocol}.txt", "w", encoding="utf-8") as f:
            try:
                soup = BeautifulSoup(requests.get(url).text, "html.parser")

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


