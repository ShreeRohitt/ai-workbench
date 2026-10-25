import pytest

from rag.chunking import chunk_by_heading, chunk_fixed
from rag.tokens import count_tokens
from rag.types import Doc

MAX, OVERLAP, MIN = 60, 10, 20

FILLER = "Team members may take this leave after notifying their manager in Workday. "

TEXT = f"""Intro paragraph before any heading. It explains what this page covers and who it
applies to, so readers know whether to keep going.

## Leave

General leave text that sits under the Leave heading before any subsection begins,
long enough that it is not considered a tiny section.

### Parental leave

{FILLER * 12}

### Sick leave

Short.

### Bereavement

Also short.

## Code example

```bash
# this is a comment, not a heading
echo "hello"
```

## Zürich office — notes

Unicode text: café, naïve, 東京, emoji 🎉🎉, em-dash — done.
"""

DOC = Doc(doc_id="people-group/leave.md", title="Leave Policy", text=TEXT)


def assert_exact_slices(chunks, doc):
    for c in chunks:
        assert c.doc_id == doc.doc_id
        assert c.text == doc.text[c.start_char : c.end_char]
        assert c.text.strip(), "empty chunk"
        assert c.n_tokens == count_tokens(c.text)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


# ---------- fixed-size ----------


@pytest.fixture
def fixed():
    return chunk_fixed(DOC, max_tokens=MAX, overlap=OVERLAP)


def test_fixed_chunks_are_exact_nonempty_slices(fixed):
    assert len(fixed) > 1
    assert_exact_slices(fixed, DOC)


def test_fixed_respects_max_tokens(fixed):
    assert all(c.n_tokens <= MAX for c in fixed)


def test_fixed_loses_no_text(fixed):
    assert fixed[0].start_char == 0
    assert fixed[-1].end_char == len(DOC.text)
    for a, b in zip(fixed, fixed[1:], strict=False):
        assert b.start_char < a.end_char, "gap between consecutive chunks"


def test_fixed_overlap_is_exact(fixed):
    for a, b in zip(fixed, fixed[1:], strict=False):
        assert count_tokens(DOC.text[b.start_char : a.end_char]) == OVERLAP


def test_fixed_has_no_heading_path(fixed):
    assert all(c.heading_path == () for c in fixed)


def test_fixed_short_doc_is_one_chunk():
    doc = Doc(doc_id="x.md", title="X", text="Just a sentence.")
    [c] = chunk_fixed(doc, max_tokens=MAX, overlap=OVERLAP)
    assert c.text == doc.text


@pytest.mark.parametrize("overlap", [-1, MAX, MAX + 5])
def test_fixed_rejects_bad_overlap(overlap):
    with pytest.raises(ValueError):
        chunk_fixed(DOC, max_tokens=MAX, overlap=overlap)


# ---------- heading-aware ----------


@pytest.fixture
def by_heading():
    return chunk_by_heading(DOC, max_tokens=MAX, min_tokens=MIN)


def chunks_containing(chunks, needle):
    return [c for c in chunks if needle in c.text]


def test_heading_chunks_are_exact_nonempty_slices(by_heading):
    assert_exact_slices(by_heading, DOC)


def test_heading_respects_max_tokens(by_heading):
    assert all(c.n_tokens <= MAX for c in by_heading)


def test_heading_loses_no_text_and_does_not_overlap(by_heading):
    ordered = sorted(by_heading, key=lambda c: c.start_char)
    edges = [0] + [x for c in ordered for x in (c.start_char, c.end_char)] + [len(DOC.text)]
    for gap_start, gap_end in zip(edges[::2], edges[1::2], strict=True):
        assert gap_start <= gap_end, "chunks overlap"
        assert not DOC.text[gap_start:gap_end].strip(), "non-whitespace text lost"


def test_heading_path_starts_with_title(by_heading):
    assert all(c.heading_path[0] == "Leave Policy" for c in by_heading)


def test_intro_before_first_heading_gets_title_path(by_heading):
    [c] = chunks_containing(by_heading, "Intro paragraph")
    assert c.heading_path == ("Leave Policy",)


def test_long_section_is_split_and_keeps_path(by_heading):
    parental = chunks_containing(by_heading, "Workday")
    assert len(parental) > 1
    assert all(c.heading_path == ("Leave Policy", "Leave", "Parental leave") for c in parental)


def test_tiny_siblings_are_merged_under_parent_path(by_heading):
    [c] = chunks_containing(by_heading, "### Sick leave")
    assert "### Bereavement" in c.text
    assert c.heading_path == ("Leave Policy", "Leave")


def test_hash_inside_code_block_is_not_a_heading(by_heading):
    assert not any("comment" in h for c in by_heading for h in c.heading_path)
    assert chunks_containing(by_heading, "# this is a comment")
