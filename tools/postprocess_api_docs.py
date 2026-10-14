"""Fix lazydocs output so that it renders correctly as Markdown.

lazydocs does not understand two things used in the docstrings:

* Sphinx roles such as ``:meth:`fork` `` are turned into plain inline code.
* ``Example:`` sections written as doctests (``>>>``) are mangled; they are
  rebuilt as fenced ``python`` blocks.

Usage: python tools/postprocess_api_docs.py [docs/api]
"""
import re
import sys
from pathlib import Path

ROLE = re.compile(r":(?:py:)?(?:meth|func|class|mod|attr|data|exc|obj|ref):`~?\.?([^`]+)`")
FIRST = re.compile(r"^\s*``` ?(.*?)```$")


def fix_example(lines: list[str]) -> list[str]:
    """Rebuild the body of one ``**Example:**`` block (lines after the header)."""
    body = []
    for i, line in enumerate(lines):
        m = FIRST.match(line) if i == 0 else None
        body.append((">>> " + m.group(1)) if m else line[4:] if line.startswith("    ") else line.strip())
    chunks, cur = [], []
    for line in body + [""]:
        if line.strip():
            cur.append(line)
        elif cur:
            chunks.append(cur)
            cur = []
    out = []
    for chunk in chunks:
        if chunk[0].startswith(">>>"):
            out += ["```python", *chunk, "```", ""]
        else:
            out += [*chunk, ""]
    return out


def process(text: str) -> str:
    text = ROLE.sub(lambda m: f"`{m.group(1)}`", text)
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):
        out.append(lines[i])
        if lines[i] == "**Example:**":
            j = i + 1
            while j < len(lines) and not (lines[j] == "---" or lines[j].startswith(("**", "#", "<a "))):
                j += 1
            block = lines[i + 1 : j]
            while block and not block[-1].strip():
                block.pop()
            out += fix_example(block) + [""]
            i = j
            continue
        i += 1
    return "\n".join(out)


if __name__ == "__main__":
    for path in Path(sys.argv[1] if len(sys.argv) > 1 else "docs/api").glob("*.md"):
        path.write_text(process(path.read_text(encoding="utf-8")), encoding="utf-8")
