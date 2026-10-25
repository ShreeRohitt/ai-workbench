"""Load handbook Markdown into Docs. YOU implement `load_docs`."""

from __future__ import annotations

from pathlib import Path

from rag.types import Doc


def load_docs(root: Path) -> list[Doc]:
    """Read every *.md file under `root` (recursively) into a Doc, sorted by doc_id.

    Contract (see tests/test_ingest.py):
    - doc_id is the path relative to `root`, POSIX separators.
    - Front-matter (the leading `---` ... `---` block) is removed from text;
      its `title:` becomes Doc.title, falling back to the file stem.
    - Hugo shortcode tags like `{{% details summary="x" %}}`, `{{% /details %}}`,
      `{{< youtube abc >}}` are removed; the content *between* paired tags is kept.
    - Docs whose text is empty/whitespace after cleaning are skipped.
    - Non-.md files are ignored.
    """
    raise NotImplementedError
