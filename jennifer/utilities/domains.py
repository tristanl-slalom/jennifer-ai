def extract_domain(domain: str):
    skip = domain.index("://") + 3
    return domain[skip:].strip("/")
