from urllib.parse import urlparse


def extract_domain(url: str):
    """Extracts the domain from the given URL, such as https://www.cnn.com -> www.cnn.com"""
    return urlparse(url).netloc
