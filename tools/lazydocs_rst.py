"""Generate one API page per listed module, with reStructuredText cleanup.

lazydocs parses Google-style sections line by line and reads any line with a
colon as a new argument. A Sphinx role such as ``:class:`Foo``` on a wrapped
line is therefore mistaken for an argument, which breaks the rest of the
section (stray bold text, ``: class:`` fragments). Rewriting roles and double
backticks to plain Markdown code before lazydocs sees the docstring avoids it.

Package names document their __init__.py without walking submodules.
"""

import argparse
import re
from importlib import import_module
from pathlib import Path

from lazydocs import generation

ROLE = re.compile(r":(?:py:)?(?:meth|func|class|mod|attr|data|exc|obj|ref):\s*`~?\.?([^`]+)`")
LITERAL = re.compile(r"``([^`\n]+?)``")

_get_docstring = generation._get_docstring


def _markdown_docstring(obj) -> str:
    doc = _get_docstring(obj)
    return LITERAL.sub(r"`\1`", ROLE.sub(r"`\1`", doc))


generation._get_docstring = _markdown_docstring


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, default=Path("docs"))
    parser.add_argument("--overview-file")
    parser.add_argument("--src-base-url")
    parser.add_argument("modules", nargs="+")
    args = parser.parse_args()
    args.output_path.mkdir(parents=True, exist_ok=True)

    generator = generation.MarkdownGenerator(
        src_root_path=str(Path(__file__).resolve().parents[1]),
        src_base_url=args.src_base_url,
        remove_package_prefix=True,
    )
    for name in dict.fromkeys(args.modules):
        print(f"Generating docs for python import: {name}")
        page = generator.module2md(import_module(name))
        if not page:
            raise ValueError(f"No API documentation generated for {name}.")
        generation.to_md_file(page, name, out_path=args.output_path)

    if args.overview_file:
        generation.to_md_file(generator.overview2md(), args.overview_file, out_path=args.output_path)


if __name__ == "__main__":
    main()
