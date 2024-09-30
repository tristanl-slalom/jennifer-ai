from html.parser import HTMLParser


class HyperlinkParser(HTMLParser):
    hyperlinks: set = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        if tag == "a" and "href" in attrs:
            self.hyperlinks.add(attrs["href"])
