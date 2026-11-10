import re
import unicodedata

_TOKEN = re.compile(r"\w+|[^\w\s]")
_WHITESPACE = re.compile(r"\s+")


def count_tokens(text: str) -> int:
    return len(_TOKEN.findall(text))


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return _WHITESPACE.sub(" ", text).strip()
