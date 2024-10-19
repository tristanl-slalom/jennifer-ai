from collections import deque
from pathlib import Path
from shutil import rmtree

import requests
from bs4 import BeautifulSoup
from rich.progress import Progress

from jennifer.utilities.domains import extract_domain
from jennifer.utilities.hyperlinks import get_domain_hyperlinks


def crawl_action(url: str, rebuild: bool, must_include: str):
    local_domain = extract_domain(url)

    # Create a queue with the original URL
    queue = deque([url])

    # We want to avoid scraping the same URL twice, so we'll also keep a running tab
    # of what we've done.
    seen = {url}

    # The scraped data will be sent to a text folder specific to the domain, so we can
    # keep reusing the same data between commands.
    output_path = Path("output")
    text_domain_dir = output_path / "text" / local_domain

    # Abort if we already have any data for the domain, and we're not rebuilding. If we
    # are rebuilding and the data exists, delete it all.
    if text_domain_dir.exists():
        if not rebuild:
            return
        else:
            rmtree(text_domain_dir)

    text_domain_dir.mkdir(exist_ok=True, parents=True)
    print(
        f"Scraping sites from domain {local_domain}. You can cancel this process at anytime with Ctrl-C.\n"
        "If you abort and re-run, you will proceed onto other steps leveraging what data you scraped before.\n"
        "Re-run the command with the --rebuild flag to clear your local data and try again.\n"
        "NOTE: additional URLs are gathered as the scraping process proceeds; the progress bar will fluctuate!\n"
    )

    with Progress() as progress:
        task = progress.add_task(f"Scraping websites for {local_domain}...", total=len(queue))
        while queue:
            url = queue.pop()
            # We currently have to do everything we've seen, minus what we've accomplished.
            # We've accomplished what we've seen minus what's left in the queue. Keep in
            # mind, every page we scrape may grow the queue further!
            progress.update(task, total=len(seen), completed=len(seen) - len(queue))

            # We need a file unique enough to reflect each page we're scraping.
            sanitized_url = url[8:].replace("/", "__").replace("?", "__").replace(":", "--")[:64]
            with open(text_domain_dir / f"{sanitized_url}.txt", "w", encoding="utf-8") as f:
                try:
                    # The user agent has to be something or sites will detect we're a robot
                    # although what the agent has to be doesn't seem to matter.
                    raw_page = requests.get(url, headers={"User-Agent": "XY"})
                    soup = BeautifulSoup(raw_page.text, "html.parser")
                    text = soup.get_text()
                    f.write(text)
                except requests.exceptions.ConnectionError:
                    # Just keep swimming.
                    continue

            for link in get_domain_hyperlinks(local_domain, url):
                if link not in seen and (not must_include or must_include in link):
                    queue.append(link)
                    seen.add(link)
