"""Fix lazydocs output so that it renders correctly as Markdown.

lazydocs does not understand two things used in the docstrings:

* Sphinx roles such as ``:meth:`fork` `` are turned into plain inline code.
* ``Example:`` sections written as doctests (``>>>``) are mangled; they are
  rebuilt as fenced ``python`` blocks of plain code, without prompts, with
  doctest outputs shown as comments.

Usage: python tools/postprocess_api_docs.py [docs/api]
"""
import re
import sys
from pathlib import Path

ROLE = re.compile(r":(?:py:)?(?:meth|func|class|mod|attr|data|exc|obj|ref):`~?\.?([^`]+)`")
FIRST = re.compile(r"^\s*``` ?(.*?)```$")


def _to_script(chunk: list[str]) -> list[str]:
    """Turn a doctest chunk into plain Python: drop the prompts, show outputs as comments."""
    code = []
    for line in chunk:
        if line.startswith((">>> ", "... ")) or line in (">>>", "..."):
            code.append(line[4:])
        elif code and "#" not in code[-1] and len(code[-1]) + len(line) <= 92:
            code[-1] += f"  # {line.strip()}"
        else:
            code.append(f"# {line.strip()}")
    return code


def fix_example(lines: list[str]) -> list[str]:
    """Rebuild the body of one ``**Example:**`` block (lines after the header)."""
    m = FIRST.match(lines[0]) if lines else None
    head, rest = ([">>> " + m.group(1)], lines[1:]) if m else ([], lines)
    indents = [len(line) - len(line.lstrip()) for line in rest if line.strip()]
    cut = min(indents, default=0)
    body = head + [line[cut:].rstrip() if line.strip() else "" for line in rest]
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
            out += ["```python", *_to_script(chunk), "```", ""]
        else:
            out += [" ".join(line.strip() for line in chunk), ""]
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
