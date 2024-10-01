def extract_domain(domain):
    skip = domain.index("://") + 3
    return domain[skip:].strip("/")
