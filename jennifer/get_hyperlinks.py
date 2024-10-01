from urllib import request
from urllib.request import Request

from jennifer.hyperlink_parser import HyperlinkParser


def get_hyperlinks(url):
    try:
        req = Request(url)
        req.add_header("User-Agent", "XY")
        with request.urlopen(req) as response:
            if not response.info().get("Content-Type").startswith("text/html"):
                return []

            html = response.read().decode("utf-8")
    except Exception as e:
        print(e)
        return []

    parser = HyperlinkParser()
    parser.feed(html)

    return parser.hyperlinks
