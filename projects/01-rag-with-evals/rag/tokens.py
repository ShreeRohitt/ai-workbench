"""Token counting. o200k_base is a proxy until the M2 embedding model is chosen."""

from __future__ import annotations

from functools import cache

import tiktoken

ENCODING = "o200k_base"


@cache
def encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING)


def encode(text: str) -> list[int]:
    # Handbook text may literally contain "<|endoftext|>"; treat it as plain text.
    return encoder().encode(text, disallowed_special=())


def count_tokens(text: str) -> int:
    return len(encode(text))
