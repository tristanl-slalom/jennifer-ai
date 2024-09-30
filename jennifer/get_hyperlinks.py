from urllib import request

from jennifer.hyperlink_parser import HyperlinkParser


def get_hyperlinks(url):
    try:
        with request.urlopen(url) as response:
            if not response.info().get('Content-Type').startswith("text/html"):
                return []

            html = response.read().decode('utf-8')
    except Exception as e:
        print(e)
        return []

    parser = HyperlinkParser()
    parser.feed(html)

    return parser.hyperlinks
