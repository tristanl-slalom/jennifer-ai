from html.parser import HTMLParser
from re import search
from typing import Set, Optional
from urllib import request
from urllib.error import HTTPError
from urllib.request import Request

from jennifer.utilities.domains import extract_domain

HTTP_URL_PATTERN = r"^http[s]*://.+"


class HyperlinkParser(HTMLParser):
    """
    Today I learned there was an HTMLParser class.
    """

    # The hyperlinks we detected when reading the HTML.
    hyperlinks: set = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        if tag == "a" and "href" in attrs:
            self.hyperlinks.add(attrs["href"])


def _read_page_for_hyperlinks(url: str) -> Set[str]:
    """
    Read a given page for it's tasty sausage links.
    """
    try:
        req = Request(url)
        # Again, set _some_ user agent or pages we'll think we're robots.
        # We're hackers now!
        req.add_header("User-Agent", "XY")
        with request.urlopen(req) as response:
            if not response.info().get("Content-Type").startswith("text/html"):
                return set()  # we ain't reading links out of images here.

            html = response.read().decode("utf-8")
            parser = HyperlinkParser()
            parser.feed(html)
            return parser.hyperlinks
    except HTTPError:
        return set()


def get_domain_hyperlinks(local_domain: str, url: str) -> Set[str]:
    """
    Load the given URL and look for links. Any unique links on the page will
    be returned in a list.
    """
    clean_links = set()

    for link in _read_page_for_hyperlinks(url):
        # If the link turned into a "clean" link, add it to the list of links.
        # If that _clean_link function decided a link was "unclean", we discard
        # it like the sinner it is. Shame!
        if clean_link := _clean_link(link, local_domain):
            clean_links.add(clean_link)

    return clean_links


def _clean_link(link: str, filtered_domain: str) -> Optional[str]:
    """
    Fixes links that are relative to be absolute, and removes URLs from
    other domains, otherwise we might be here all day.
    """
    clean_link = None
    if search(HTTP_URL_PATTERN, link):
        link_domain = extract_domain(link)
        if link_domain == filtered_domain:
            clean_link = link
    else:
        if link.startswith("//"):
            # MacRumors has links to YouTube like "//www.youtube.com/@themacrumorsshow"
            # Like... is that a thing?
            return None
        elif link.startswith("#") or link.startswith("mailto:"):
            return None
        elif link.startswith("/"):
            # A relative link that starts with a slash should be fixed by removing the slash
            # and then reassembled.
            link = link[1:]
            clean_link = "https://" + filtered_domain + "/" + link

    if clean_link and clean_link.endswith("/"):
        # Remove trailing slashes just because I guess, I don't know.
        clean_link = clean_link[:-1]

    return clean_link
