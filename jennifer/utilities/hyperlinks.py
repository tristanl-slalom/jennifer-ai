from html.parser import HTMLParser
from re import search
from urllib import request
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request

HTTP_URL_PATTERN = r"^http[s]*://.+"


class HyperlinkParser(HTMLParser):
    hyperlinks: set = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        if tag == "a" and "href" in attrs:
            self.hyperlinks.add(attrs["href"])


def get_hyperlinks(url):
    try:
        req = Request(url)
        req.add_header("User-Agent", "XY")
        with request.urlopen(req) as response:
            if not response.info().get("Content-Type").startswith("text/html"):
                return []

            html = response.read().decode("utf-8")
    except HTTPError:
        return []

    parser = HyperlinkParser()
    parser.feed(html)

    return parser.hyperlinks


def get_domain_hyperlinks(local_domain: str, url: str):
    clean_links = set()
    for link in get_hyperlinks(url):
        if link.startswith("frm") and url.endswith(".com/"):
            clean_links.add(f"{url}{link}")
        elif url.startswith("https://") and "/frm" in url:
            clean_links.add(url)
        elif clean_link := _clean_link(link, local_domain):
            clean_links.add(clean_link)

    return list(clean_links)


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
