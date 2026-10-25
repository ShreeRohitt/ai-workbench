from rag.ingest import load_docs

PAGE = """---
title: Parental Leave
description: How parental leave works
---

## Eligibility

All team members are eligible.

{{% details summary="Country specifics" %}}
In Germany, statutory rules also apply.
{{% /details %}}

{{< youtube abc123 >}}
"""


def write(root, rel, content):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_load_docs(tmp_path):
    write(tmp_path, "people-group/parental.md", PAGE)
    write(tmp_path, "people-group/_index.md", "## No front-matter here\n\nBody.\n")
    write(tmp_path, "people-group/empty.md", "---\ntitle: Empty\n---\n\n  \n")
    write(tmp_path, "people-group/image.png", "not markdown")

    docs = load_docs(tmp_path)

    assert [d.doc_id for d in docs] == ["people-group/_index.md", "people-group/parental.md"]
    index, parental = docs
    assert index.title == "_index"
    assert parental.title == "Parental Leave"
    assert "description:" not in parental.text
    assert not parental.text.lstrip().startswith("---")
    assert "{{" not in parental.text and "}}" not in parental.text
    assert "In Germany, statutory rules also apply." in parental.text
    assert "## Eligibility" in parental.text
