from collections import deque
from pathlib import Path
from shutil import rmtree

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel
from rich.progress import Progress

from jennifer.utilities.domains import extract_domain, url_to_filename
from jennifer.utilities.hyperlinks import get_domain_hyperlinks


class CrawlMetadata(BaseModel):
    local_domain: str
    output_path: Path
    text_domain_dir: Path


def crawl_metadata_from_url(url: str):
    local_domain = extract_domain(url)
    output_path = Path("output")
    text_domain_dir = output_path / "text" / local_domain
    return CrawlMetadata(local_domain=local_domain, output_path=output_path, text_domain_dir=text_domain_dir)


def crawl_action(url: str, rebuild: bool, must_include: str) -> CrawlMetadata:
    metadata = crawl_metadata_from_url(url)

    # Create a queue with the original URL
    queue = deque([url])

    # We want to avoid scraping the same URL twice, so we'll also keep a running tab
    # of what we've done.
    seen = {url}

    # The scraped data will be sent to a text folder specific to the domain, so we can
    # keep reusing the same data between commands.
    # Abort if we already have any data for the domain, and we're not rebuilding. If we
    # are rebuilding and the data exists, delete it all.
    if metadata.text_domain_dir.exists():
        if not rebuild:
            return metadata
        else:
            rmtree(metadata.text_domain_dir)

    metadata.text_domain_dir.mkdir(exist_ok=True, parents=True)
    print(
        f"Scraping sites from domain {metadata.local_domain}. You can cancel this process at anytime with Ctrl-C.\n"
        "If you abort and re-run, you will proceed onto other steps leveraging what data you scraped before.\n"
        "Re-run the command with the --rebuild flag to clear your local data and try again.\n"
        "NOTE: additional URLs are gathered as the scraping process proceeds; the progress bar will fluctuate!\n"
    )

    with Progress() as progress:
        task = progress.add_task(f"Scraping websites for {metadata.local_domain}...", total=len(queue))
        while queue:
            url = queue.pop()
            # We currently have to do everything we've seen, minus what we've accomplished.
            # We've accomplished what we've seen minus what's left in the queue. Keep in
            # mind, every page we scrape may grow the queue further!
            progress.update(task, total=len(seen), completed=len(seen) - len(queue))

            # We need a file unique enough to reflect each page we're scraping.
            url_filename = url_to_filename(url)
            with open(metadata.text_domain_dir / f"{url_filename}.txt", "w", encoding="utf-8") as f:
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

            for link in get_domain_hyperlinks(metadata.local_domain, url):
                if link not in seen and (not must_include or must_include in link):
                    queue.append(link)
                    seen.add(link)

    return metadata
