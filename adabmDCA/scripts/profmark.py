"""Compatibility shim for the former module name."""

from adabmDCA.scripts.split_data import create_parser, main, run  # noqa: F401  (re-exported)


if __name__ == "__main__":
    raise SystemExit(main())
