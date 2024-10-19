import re
from urllib.parse import urlparse


def extract_domain(url: str):
    """Extracts the domain from the given URL, such as https://www.cnn.com -> www.cnn.com"""
    return urlparse(url).netloc


def url_to_filename(url: str) -> str:
    """Converts an absolute URL to a filename, replacing any invalid file path characters with __"""
    domain = extract_domain(url)

    if "://" in url:
        url = url[url.index("://") + 3 :]

    # Replace all file-system breaking characters with dunderscore and only
    # keep up to the first 64 characters.
    sanitized_url = re.sub(r'[\\/:*?"<>|]', "__", url[len(domain) :])[:64]
    if not sanitized_url:
        return domain

    return sanitized_url
