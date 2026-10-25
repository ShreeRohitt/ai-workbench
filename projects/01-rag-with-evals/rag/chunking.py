"""Two chunking strategies over a Doc. YOU implement both functions."""

from __future__ import annotations

from rag.types import Chunk, Doc


def chunk_fixed(doc: Doc, max_tokens: int = 512, overlap: int = 64) -> list[Chunk]:
    """Sliding window over the doc's tokens.

    Contract (see tests/test_chunking.py):
    - Each chunk has <= max_tokens tokens; consecutive chunks share exactly `overlap` tokens.
    - Chunks cover the whole doc: first starts at 0, last ends at len(doc.text).
    - chunk.text == doc.text[start_char:end_char]; heading_path is ().
    - Raise ValueError unless 0 <= overlap < max_tokens.

    Hint: you need token -> character offsets. Look at `encoder().decode_with_offsets`
    in rag/tokens.py's encoder, and think about what happens when a token boundary
    falls inside a multi-byte character (the fixture has emoji and CJK on purpose).
    """
    raise NotImplementedError


def chunk_by_heading(doc: Doc, max_tokens: int = 512, min_tokens: int = 50) -> list[Chunk]:
    """Split on Markdown headings; carry the heading path.

    Contract (see tests/test_chunking.py):
    - A section = a heading line plus everything up to the next heading of any level.
      Text before the first heading is its own section with heading_path (doc.title,).
    - heading_path = (doc.title, H1/H2..., current heading), built from the heading stack.
    - `#` lines inside fenced code blocks are NOT headings.
    - Sections > max_tokens are split further (reuse chunk_fixed's logic, no overlap needed),
      every piece keeping the section's heading_path.
    - Sections < min_tokens are merged with following *sibling* sections (same parent) while
      the merged chunk stays <= max_tokens. A merged chunk's heading_path is the parent's path.
    - Chunks don't overlap; any text between chunks is whitespace only.
    - chunk.text == doc.text[start_char:end_char].
    """
    raise NotImplementedError
