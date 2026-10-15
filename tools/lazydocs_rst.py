"""Run lazydocs after rewriting reStructuredText markup in docstrings.

lazydocs parses Google-style sections line by line and reads any line with a
colon as a new argument. A Sphinx role such as ``:class:`Foo``` on a wrapped
line is therefore mistaken for an argument, which breaks the rest of the
section (stray bold text, ``: class:`` fragments). Rewriting roles and double
backticks to plain Markdown code before lazydocs sees the docstring avoids it.

Usage: python tools/lazydocs_rst.py [lazydocs options] MODULE...
"""
import re

from lazydocs import generation
from lazydocs._cli import app

ROLE = re.compile(r":(?:py:)?(?:meth|func|class|mod|attr|data|exc|obj|ref):\s*`~?\.?([^`]+)`")
LITERAL = re.compile(r"``([^`\n]+?)``")

_get_docstring = generation._get_docstring


def _markdown_docstring(obj) -> str:
    doc = _get_docstring(obj)
    return LITERAL.sub(r"`\1`", ROLE.sub(r"`\1`", doc))


generation._get_docstring = _markdown_docstring

if __name__ == "__main__":
    app(prog_name="lazydocs")
