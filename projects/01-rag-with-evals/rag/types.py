"""Core records passed between ingestion, chunking and retrieval."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Doc:
    doc_id: str  # path relative to the handbook root, POSIX style, e.g. "people-group/visas.md"
    title: str  # from front-matter `title:`, else the file stem
    text: str  # Markdown body: front-matter removed, shortcode tags removed (inner content kept)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # f"{doc_id}::{strategy}::{index}", unique and deterministic
    doc_id: str
    text: str  # always exactly doc.text[start_char:end_char]
    start_char: int
    end_char: int
    n_tokens: int  # count_tokens(text)
    heading_path: tuple[str, ...] = ()  # (doc title, H2, H3, ...); empty for fixed-size chunks
