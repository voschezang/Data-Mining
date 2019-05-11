from termcolor import colored


def print_primary(*args):
    print(colored(*args, 'green'))


def print_secondary(*args):
    print(colored(*args, 'blue'))


def print_warning(*args):
    print(colored(*args, 'red'))


def title(text: str) -> str:
    print(text)
    if all(c.isdigit() or c.isupper() for c in text):
        return text
    return str.upper(text)


def pad_labels(keys, n=None):
    if n is None:
        n = max(len(k) for k in keys)
    return [k.center(n) for k in keys]


def select_if_contains(strings, substrings) -> list:
    selection = []
    for s in substrings:
        selection.extend([k for k in strings if s in k])
    return selection


def remove(strings, selection):
    for s in selection:
        if s in strings:
            strings.remove(s)
    # return [k for k in strings if k not in selection]
