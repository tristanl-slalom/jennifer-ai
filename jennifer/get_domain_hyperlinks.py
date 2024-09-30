from re import search
from urllib.parse import urlparse

from jennifer.get_hyperlinks import get_hyperlinks


HTTP_URL_PATTERN = r'^http[s]*://.+'


def _clean_link(link: str, local_domain: str):
    clean_link = None
    if search(HTTP_URL_PATTERN, link):
        url_obj = urlparse(link)
        if url_obj.netloc == local_domain:
            clean_link = link
    else:
        if link.startswith("/"):
            link = link[1:]
        elif link.startswith("#") or link.startswith("mailto:"):
            return None
        clean_link = "https://" + local_domain + "/" + link

    if clean_link is not None:
        if clean_link.endswith("/"):
            clean_link = clean_link[:-1]

    return clean_link


def get_domain_hyperlinks(local_domain: str, url: str):
    clean_links = set()
    for link in get_hyperlinks(url):
        if link.startswith('frm') and url.endswith(".com/"):
            clean_links.add(f"{url}{link}")
        elif clean_link := _clean_link(link, local_domain):
            clean_links.add(clean_link)

    return list(clean_links)
