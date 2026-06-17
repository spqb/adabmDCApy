from pathlib import Path


def ensure_output_dir(path: str | Path) -> Path:
    """Create an output directory and return it as a Path."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def require_file(path: str | Path, description: str) -> Path:
    """Validate that a required input file exists."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{description} {file_path} not found.")
    if not file_path.is_file():
        raise FileNotFoundError(f"{description} {file_path} is not a file.")
    return file_path


def prefixed_path(output_dir: str | Path, label: str | None, default_stem: str) -> Path:
    stem = f"{label}_{default_stem}" if label is not None else default_stem
    return Path(output_dir) / stem
