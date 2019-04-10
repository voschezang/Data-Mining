
def title(text: str) -> str:
    print(text)
    if all(c.isdigit() or c.isupper() for c in text):
        return text
    return str.upper(text)


def pad_labels(keys, n=None):
    if n is None:
        n = max(len(k) for k in keys)
    return [k.center(n) for k in keys]
