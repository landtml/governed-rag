from __future__ import annotations

import re
from collections.abc import Iterator

from governed_rag.text import count_tokens
from governed_rag.types import Chunk

_HEADING = re.compile(r"(#{1,6}) ")
_LIST = re.compile(r"(?:[-*]|\d+\.) ")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _is_block_start(line: str) -> bool:
    return line.startswith(("```", "|")) or bool(_HEADING.match(line) or _LIST.match(line))


def _blocks(body: str) -> Iterator[tuple]:
    """Yield ("heading", level, title) or ("block", kind, text) in document order."""
    lines = body.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
        elif line.startswith("```"):
            # An unclosed fence runs to the end of the body; its contents are opaque.
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            end = min(j + 1, len(lines))
            yield ("block", "code", "\n".join(lines[i:end]))
            i = end
        elif (m := _HEADING.match(line)):
            yield ("heading", len(m.group(1)), line[m.end():].strip())
            i += 1
        elif line.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            yield ("block", "table", "\n".join(lines[i:j]))
            i = j
        elif _LIST.match(line):
            # A list runs on through blank lines while the next non-blank line is an item or an
            # indented continuation. Trailing blank lines after the last item are not part of it.
            j = i + 1          # exclusive end of the block so far
            k = i + 1          # scan position
            while k < len(lines):
                if not lines[k].strip():
                    k += 1
                elif _LIST.match(lines[k]) or lines[k][0] in " \t":
                    k += 1
                    j = k
                else:
                    break
            yield ("block", "list", "\n".join(lines[i:j]))
            i = j
        else:
            # A block-start line also ends a paragraph, so a heading or fence directly
            # under paragraph text still opens its own block.
            j = i + 1
            while j < len(lines) and lines[j].strip() and not _is_block_start(lines[j]):
                j += 1
            yield ("block", "paragraph", "\n".join(lines[i:j]).strip())
            i = j


def _pack_sentences(text: str, target: int) -> list[str]:
    packs: list[str] = []
    current: list[str] = []
    total = 0
    for sentence in _SENTENCE.split(text):
        n = count_tokens(sentence)
        if current and total + n > target:
            packs.append(" ".join(current))
            current, total = [], 0
        current.append(sentence)
        total += n
    if current:
        packs.append(" ".join(current))
    return packs


def chunk_markdown(title: str, body: str, target: int, max_tokens: int) -> list[Chunk]:
    pieces: list[tuple[str, str]] = []  # (section, body)
    stack: list[tuple[int, str]] = []
    section = ""
    current: list[str] = []
    total = 0

    def close() -> None:
        nonlocal current, total
        if current:
            pieces.append((section, "\n\n".join(current)))
        current, total = [], 0

    for event in _blocks(body):
        if event[0] == "heading":
            _, level, text = event
            close()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            section = " > ".join(t for _, t in stack)
            continue

        _, kind, text = event
        n = count_tokens(text)
        if kind == "paragraph" and n > max_tokens:
            close()
            pieces.extend((section, pack) for pack in _pack_sentences(text, target))
            continue
        if current and total + n > target:
            close()
        current.append(text)
        total += n
    close()

    chunks = []
    for ord_, (sec, text) in enumerate(pieces):
        head = f"{title} > {sec}" if sec else title
        chunks.append(Chunk(ord=ord_, section=sec, body=text, index_text=f"{head}\n\n{text}"))
    return chunks
